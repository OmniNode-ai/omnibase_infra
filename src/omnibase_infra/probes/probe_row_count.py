# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Row count probe for omnidash projection tables (OMN-5653).

Queries pg_stat_user_tables to detect tables with zero live rows,
which indicates projection failures (events consumed but not written).

The probe is designed to be called periodically by the runtime health
pipeline or begin-day diagnostics. It emits a diagnostic event when
empty tables are detected.

Usage:
    from omnibase_infra.probes.probe_row_count import RowCountProbe

    probe = RowCountProbe(dsn="postgresql://...")
    result = await probe.run()
    if not result.healthy:
        # emit diagnostic event
        ...
"""

from __future__ import annotations

import logging

import asyncpg

from omnibase_infra.models.health.model_row_count_probe_result import (
    ModelRowCountProbeResult,
)
from omnibase_infra.models.health.model_table_row_count import (
    ModelTableRowCount,
)

logger = logging.getLogger(__name__)

# Tables that are expected to be empty in normal operation (e.g. DLQ, staging).
# These are excluded from the "unhealthy" determination.
DEFAULT_EXPECTED_EMPTY: frozenset[str] = frozenset(
    {
        "schema_migrations",
        "drizzle_migrations",
    }
)

_ROW_COUNT_QUERY = """
    SELECT
        schemaname,
        relname,
        n_live_tup
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY relname
"""


class PgStatRow:
    """Typed wrapper for pg_stat_user_tables query results."""

    __slots__ = ("n_live_tup", "relname", "schemaname")

    def __init__(self, schemaname: str, relname: str, n_live_tup: int) -> None:
        self.schemaname = schemaname
        self.relname = relname
        self.n_live_tup = n_live_tup

    @classmethod
    def from_record(cls, record: asyncpg.Record) -> PgStatRow:
        """Construct from an asyncpg Record."""
        return cls(
            schemaname=str(record["schemaname"]),
            relname=str(record["relname"]),
            n_live_tup=int(record["n_live_tup"]),
        )


class RowCountProbe:
    """Probes a PostgreSQL database for tables with zero live rows.

    Connects via asyncpg, queries pg_stat_user_tables, and returns
    a structured result with per-table row counts.

    Args:
        dsn: PostgreSQL connection string.
        expected_empty: Table names that are expected to be empty
            (excluded from the healthy/unhealthy determination).
        db_display_label: Display name for the database being probed.
    """

    def __init__(
        self,
        *,
        dsn: str,
        expected_empty: frozenset[str] | None = None,
        db_display_label: str = "omnidash_analytics",
    ) -> None:
        self._dsn = dsn
        self._expected_empty = expected_empty or DEFAULT_EXPECTED_EMPTY
        self._db_display_label = db_display_label

    async def run(self) -> ModelRowCountProbeResult:
        """Execute the row count probe.

        Returns:
            ModelRowCountProbeResult with per-table details and health status.
        """
        rows = await self._query_row_counts()
        return self._evaluate(rows)

    async def _query_row_counts(self) -> list[PgStatRow]:
        """Query pg_stat_user_tables for row counts."""
        conn: asyncpg.Connection[asyncpg.Record] = await asyncpg.connect(self._dsn)
        try:
            records = await conn.fetch(_ROW_COUNT_QUERY)
            return [PgStatRow.from_record(r) for r in records]
        finally:
            await conn.close()

    def _evaluate(self, rows: list[PgStatRow]) -> ModelRowCountProbeResult:
        """Evaluate row counts and build the probe result."""
        table_details: list[ModelTableRowCount] = []
        empty_table_names: list[str] = []

        for row in rows:
            is_empty = row.n_live_tup == 0

            table_details.append(
                ModelTableRowCount(
                    relation_key=row.relname,
                    schema_label=row.schemaname,
                    n_live_tup=row.n_live_tup,
                    is_empty=is_empty,
                )
            )

            if is_empty and row.relname not in self._expected_empty:
                empty_table_names.append(row.relname)

        total = len(table_details)
        empty_count = len(empty_table_names)
        populated = total - empty_count

        # Healthy if no unexpected empty tables
        healthy = empty_count == 0

        if not healthy:
            logger.warning(
                "Row count probe found %d empty projection table(s): %s",
                empty_count,
                ", ".join(empty_table_names),
            )

        return ModelRowCountProbeResult(
            db_display_label=self._db_display_label,
            total_tables=total,
            empty_tables=empty_count,
            populated_tables=populated,
            empty_table_names=empty_table_names,
            table_details=table_details,
            healthy=healthy,
        )


__all__ = [
    "RowCountProbe",
    "DEFAULT_EXPECTED_EMPTY",
]
