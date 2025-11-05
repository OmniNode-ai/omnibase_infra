"""Rate Limiter Models.

Strongly-typed models for rate limiter statistics to replace Dict[str, Any] usage.
Maintains ONEX compliance with proper field validation.
"""

from pydantic import BaseModel, Field


class ModelClientStats(BaseModel):
    """Model for rate limiter client statistics."""

    # Client Information
    client_id: str = Field(
        max_length=200,
        description="Unique client identifier",
    )

    client_type: str | None = Field(
        default=None,
        max_length=50,
        description="Type of client (api, web, mobile, etc.)",
    )

    # Request Statistics
    total_requests: int = Field(
        ge=0,
        description="Total number of requests made",
    )

    allowed_requests: int = Field(
        ge=0,
        description="Number of requests that were allowed",
    )

    blocked_requests: int = Field(
        ge=0,
        description="Number of requests that were blocked",
    )

    # Rate Information
    current_rate: float = Field(
        ge=0.0,
        description="Current request rate (requests per second)",
    )

    average_rate: float = Field(
        ge=0.0,
        description="Average request rate over monitoring period",
    )

    peak_rate: float = Field(
        ge=0.0,
        description="Peak request rate observed",
    )

    # Limit Information
    rate_limit: int = Field(
        ge=0,
        description="Current rate limit for this client",
    )

    limit_window_seconds: int = Field(
        ge=1,
        le=3600,
        description="Time window for rate limiting in seconds",
    )

    burst_limit: int | None = Field(
        default=None,
        ge=0,
        description="Burst limit for this client",
    )

    # Timing Information
    first_request_time: str = Field(
        description="ISO timestamp of first request",
    )

    last_request_time: str = Field(
        description="ISO timestamp of last request",
    )

    last_blocked_time: str | None = Field(
        default=None,
        description="ISO timestamp of last blocked request",
    )

    # Penalty Information
    penalty_count: int = Field(
        default=0,
        ge=0,
        description="Number of penalties applied",
    )

    current_penalty_expires: str | None = Field(
        default=None,
        description="ISO timestamp when current penalty expires",
    )

    total_penalty_time_seconds: int = Field(
        default=0,
        ge=0,
        description="Total time spent in penalty",
    )

    # Status Information
    is_blocked: bool = Field(
        default=False,
        description="Whether client is currently blocked",
    )

    is_whitelisted: bool = Field(
        default=False,
        description="Whether client is whitelisted",
    )

    is_blacklisted: bool = Field(
        default=False,
        description="Whether client is blacklisted",
    )

    # Additional Context
    user_agent: str | None = Field(
        default=None,
        max_length=500,
        description="User agent string (if available)",
    )

    source_ip: str | None = Field(
        default=None,
        max_length=45,
        description="Source IP address (hashed for privacy)",
    )

    geographic_region: str | None = Field(
        default=None,
        max_length=100,
        description="Geographic region of client",
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"


class ModelGlobalStats(BaseModel):
    """Model for global rate limiter statistics."""

    # Overall Request Statistics
    total_requests_all_clients: int = Field(
        ge=0,
        description="Total requests across all clients",
    )

    total_allowed_requests: int = Field(
        ge=0,
        description="Total allowed requests across all clients",
    )

    total_blocked_requests: int = Field(
        ge=0,
        description="Total blocked requests across all clients",
    )

    # Rate Statistics
    global_request_rate: float = Field(
        ge=0.0,
        description="Global request rate (requests per second)",
    )

    average_client_rate: float = Field(
        ge=0.0,
        description="Average rate per client",
    )

    peak_global_rate: float = Field(
        ge=0.0,
        description="Peak global request rate observed",
    )

    # Client Statistics
    total_active_clients: int = Field(
        ge=0,
        description="Number of active clients",
    )

    total_blocked_clients: int = Field(
        ge=0,
        description="Number of currently blocked clients",
    )

    total_whitelisted_clients: int = Field(
        default=0,
        ge=0,
        description="Number of whitelisted clients",
    )

    total_blacklisted_clients: int = Field(
        default=0,
        ge=0,
        description="Number of blacklisted clients",
    )

    # Performance Statistics
    average_processing_time_ms: float = Field(
        ge=0.0,
        description="Average processing time per request in milliseconds",
    )

    cache_hit_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Cache hit rate percentage",
    )

    memory_usage_mb: float | None = Field(
        default=None,
        ge=0.0,
        description="Memory usage in megabytes",
    )

    # Time Window Information
    statistics_window_seconds: int = Field(
        ge=1,
        description="Time window these statistics cover",
    )

    statistics_generated_at: str = Field(
        description="ISO timestamp when statistics were generated",
    )

    uptime_seconds: int = Field(
        ge=0,
        description="Rate limiter uptime in seconds",
    )

    # Configuration Information
    default_rate_limit: int = Field(
        ge=0,
        description="Default rate limit for new clients",
    )

    max_clients: int | None = Field(
        default=None,
        ge=0,
        description="Maximum number of clients supported",
    )

    cleanup_interval_seconds: int = Field(
        default=300,
        ge=60,
        description="Interval for cleanup operations",
    )

    # Health Information
    is_healthy: bool = Field(
        default=True,
        description="Whether rate limiter is operating normally",
    )

    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors encountered",
    )

    last_error_time: str | None = Field(
        default=None,
        description="ISO timestamp of last error",
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
