"""Health Status Details Model.

Strongly-typed model for additional health status details.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field


class ModelHealthDetails(BaseModel):
    """Model for additional health status details."""

    # Component-specific details
    postgres_connection_count: int | None = Field(
        default=None,
        ge=0,
        description="Current PostgreSQL connection count",
    )

    postgres_last_error: str | None = Field(
        default=None,
        max_length=500,
        description="Last PostgreSQL error message",
    )

    kafka_producer_count: int | None = Field(
        default=None,
        ge=0,
        description="Current Kafka producer count",
    )

    kafka_last_error: str | None = Field(
        default=None,
        max_length=500,
        description="Last Kafka error message",
    )

    circuit_breaker_state: str | None = Field(
        default=None,
        pattern="^(CLOSED|HALF_OPEN|OPEN)$",
        description="Current circuit breaker state",
    )

    circuit_breaker_failure_count: int | None = Field(
        default=None,
        ge=0,
        description="Circuit breaker failure count",
    )

    # Performance indicators
    peak_memory_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Peak memory usage in megabytes",
    )

    average_cpu_usage_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Average CPU usage percentage",
    )

    disk_space_available_gb: float | None = Field(
        default=None,
        ge=0.0,
        description="Available disk space in gigabytes",
    )

    # Network and connectivity
    network_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Average network latency in milliseconds",
    )

    external_service_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of external services being monitored",
    )

    external_services_healthy: int | None = Field(
        default=None,
        ge=0,
        description="Number of external services reporting healthy",
    )

    # Configuration and environment
    environment_variables_loaded: int | None = Field(
        default=None,
        ge=0,
        description="Number of environment variables loaded",
    )

    configuration_files_loaded: int | None = Field(
        default=None,
        ge=0,
        description="Number of configuration files loaded",
    )

    # Security indicators
    ssl_certificates_valid: bool | None = Field(
        default=None,
        description="Whether SSL certificates are valid",
    )

    ssl_certificates_expire_days: int | None = Field(
        default=None,
        ge=0,
        description="Days until SSL certificates expire",
    )

    # Error tracking
    recent_errors: list[str] | None = Field(
        default=None,
        max_items=10,
        description="List of recent error messages (last 10)",
    )

    warning_messages: list[str] | None = Field(
        default=None,
        max_items=10,
        description="List of recent warning messages (last 10)",
    )

    # Health check specifics
    health_check_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time taken to complete health check in milliseconds",
    )

    components_checked: int | None = Field(
        default=None,
        ge=0,
        description="Number of components included in health check",
    )

    components_healthy: int | None = Field(
        default=None,
        ge=0,
        description="Number of components reporting healthy",
    )
