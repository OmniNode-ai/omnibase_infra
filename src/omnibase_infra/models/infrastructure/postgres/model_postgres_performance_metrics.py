"""PostgreSQL performance metrics model."""


from pydantic import BaseModel, Field


class ModelPostgresPerformanceMetrics(BaseModel):
    """PostgreSQL performance metrics model."""

    queries_per_second: float | None = Field(default=None, description="Queries per second", ge=0)
    average_query_time_ms: float | None = Field(default=None, description="Average query execution time in milliseconds", ge=0)
    slow_query_count: int | None = Field(default=None, description="Number of slow queries", ge=0)
    cache_hit_ratio: float | None = Field(default=None, description="Cache hit ratio (0-1)", ge=0, le=1)
    buffer_hit_ratio: float | None = Field(default=None, description="Buffer hit ratio (0-1)", ge=0, le=1)
    disk_reads_per_second: float | None = Field(default=None, description="Disk reads per second", ge=0)
    disk_writes_per_second: float | None = Field(default=None, description="Disk writes per second", ge=0)
    cpu_usage_percent: float | None = Field(default=None, description="CPU usage percentage", ge=0, le=100)
    memory_usage_bytes: int | None = Field(default=None, description="Memory usage in bytes", ge=0)
