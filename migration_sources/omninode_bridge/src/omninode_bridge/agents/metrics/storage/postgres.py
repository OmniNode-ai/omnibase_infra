"""
PostgreSQL metrics storage with partitioned tables.

Stores metrics in partitioned tables for historical analysis.
"""

import json
import logging
from typing import ClassVar

import asyncpg

from omninode_bridge.agents.metrics.models import Metric

logger = logging.getLogger(__name__)


class PostgreSQLMetricsWriter:
    """
    PostgreSQL metrics writer with batch inserts.

    Features:
    - Batch inserts with prepared statements
    - Partitioned tables (daily partitions)
    - GIN indexes on JSONB fields
    - 90-day retention

    Performance:
    - Batch insert: <20ms for 100 metrics
    - Prepared statements: 30% faster

    Tables:
    - agent_routing_metrics
    - agent_state_metrics
    - agent_coordination_metrics
    - agent_workflow_metrics
    - agent_quorum_metrics
    """

    # Metric name prefixes to table mapping
    TABLE_MAPPING: ClassVar[dict[str, str]] = {
        "routing_": "agent_routing_metrics",
        "state_": "agent_state_metrics",
        "coordination_": "agent_coordination_metrics",
        "workflow_": "agent_workflow_metrics",
        "quorum_": "agent_quorum_metrics",
    }

    DEFAULT_TABLE = "agent_routing_metrics"

    def __init__(self, postgres_url: str, pool_size: int = 10):
        """
        Initialize PostgreSQL metrics writer.

        Args:
            postgres_url: PostgreSQL connection URL
            pool_size: Connection pool size
        """
        self._postgres_url = postgres_url
        self._pool_size = pool_size
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        """Start PostgreSQL connection pool."""
        try:
            self._pool = await asyncpg.create_pool(
                self._postgres_url,
                min_size=2,
                max_size=self._pool_size,
                command_timeout=60,
            )
            logger.info(f"PostgreSQL pool created: pool_size={self._pool_size}")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL pool: {e}")
            raise

    async def stop(self) -> None:
        """Stop PostgreSQL connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL pool closed")

    async def write_batch(self, metrics: list[Metric]) -> None:
        """
        Write batch of metrics to PostgreSQL.

        Args:
            metrics: List of metrics to write

        Performance: <20ms for 100 metrics
        """
        if not self._pool:
            logger.warning("PostgreSQL pool not started, skipping write")
            return

        if not metrics:
            return

        # Group metrics by table
        metrics_by_table = self._group_by_table(metrics)

        # Insert into tables
        async with self._pool.acquire() as conn:
            for table, table_metrics in metrics_by_table.items():
                try:
                    # Prepare batch data
                    records = [
                        (
                            metric.metric_id,
                            metric.metric_name,
                            metric.metric_type.value,
                            metric.value,
                            metric.unit,
                            json.dumps(metric.tags),  # JSONB
                            metric.agent_id,
                            metric.correlation_id,
                            metric.timestamp,
                        )
                        for metric in table_metrics
                    ]

                    # Batch insert
                    await conn.executemany(
                        f"""
                        INSERT INTO {table}
                        (metric_id, metric_name, metric_type, value, unit, tags,
                         agent_id, correlation_id, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
                        """,
                        records,
                    )

                    logger.debug(f"Inserted {len(table_metrics)} metrics into {table}")

                except asyncpg.PostgresError as e:
                    logger.error(f"PostgreSQL insert error for table {table}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error inserting to PostgreSQL: {e}")

    def _group_by_table(self, metrics: list[Metric]) -> dict[str, list[Metric]]:
        """
        Group metrics by PostgreSQL table based on metric name prefix.

        Args:
            metrics: List of metrics

        Returns:
            Dictionary mapping table to metrics list
        """
        grouped: dict[str, list[Metric]] = {}

        for metric in metrics:
            table = self._get_table_for_metric(metric.metric_name)
            if table not in grouped:
                grouped[table] = []
            grouped[table].append(metric)

        return grouped

    def _get_table_for_metric(self, metric_name: str) -> str:
        """
        Get PostgreSQL table for metric based on name prefix.

        Args:
            metric_name: Metric name

        Returns:
            Table name
        """
        for prefix, table in self.TABLE_MAPPING.items():
            if metric_name.startswith(prefix):
                return table
        return self.DEFAULT_TABLE
