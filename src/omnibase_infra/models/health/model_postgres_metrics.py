"""PostgreSQL Metrics Model.

Strongly-typed model for PostgreSQL component health metrics.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelPostgresMetrics(BaseModel):
    """Model for PostgreSQL component health metrics."""

    # Connection metrics
    active_connections: int = Field(
        ge=0,
        description="Number of active database connections"
    )

    max_connections: int = Field(
        ge=0,
        description="Maximum allowed database connections"
    )

    idle_connections: int = Field(
        ge=0,
        description="Number of idle database connections"
    )

    connection_pool_utilization: float = Field(
        ge=0.0,
        le=100.0,
        description="Connection pool utilization percentage"
    )

    # Performance metrics
    avg_query_duration_ms: float = Field(
        ge=0.0,
        description="Average query execution time in milliseconds"
    )

    slow_queries_count: int = Field(
        ge=0,
        description="Number of slow queries in the monitoring period"
    )

    queries_per_second: float = Field(
        ge=0.0,
        description="Average queries processed per second"
    )

    # Database health
    database_size_mb: float = Field(
        ge=0.0,
        description="Total database size in megabytes"
    )

    locks_count: int = Field(
        ge=0,
        description="Number of active database locks"
    )

    deadlocks_count: int = Field(
        ge=0,
        description="Number of deadlocks detected"
    )

    # Availability metrics
    uptime_seconds: int = Field(
        ge=0,
        description="Database uptime in seconds"
    )

    last_backup_timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp of last successful backup"
    )

    replication_lag_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Replication lag in milliseconds (if applicable)"
    )

    # Error metrics
    connection_errors: int = Field(
        ge=0,
        description="Number of connection errors"
    )

    transaction_rollbacks: int = Field(
        ge=0,
        description="Number of transaction rollbacks"
    )

    # Resource utilization
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Database CPU usage percentage"
    )

    memory_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Database memory usage percentage"
    )

    disk_io_read_mb_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Disk I/O read rate in MB/s"
    )

    disk_io_write_mb_s: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Disk I/O write rate in MB/s"
    )