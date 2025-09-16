"""Health Trend Analysis Model.

Strongly-typed model for health trend analysis data.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class ModelTrendDataPoint(BaseModel):
    """Model for individual trend data points."""

    timestamp: datetime = Field(
        description="Data point timestamp",
    )

    value: float = Field(
        description="Metric value at this timestamp",
    )

    metric_name: str = Field(
        description="Name of the metric being tracked",
    )


class ModelTrendAnalysis(BaseModel):
    """Model for health trend analysis data."""

    analysis_period_hours: int = Field(
        ge=1,
        description="Time period covered by this analysis in hours",
    )

    analysis_timestamp: datetime = Field(
        description="When this analysis was generated",
    )

    # Trend indicators
    overall_trend: str = Field(
        pattern="^(improving|stable|degrading|unknown)$",
        description="Overall health trend direction",
    )

    trend_confidence: float = Field(
        ge=0.0,
        le=100.0,
        description="Confidence level of trend analysis percentage",
    )

    # Performance trends
    avg_response_time_trend: float = Field(
        description="Average response time change percentage (positive = slower)",
    )

    error_rate_trend: float = Field(
        description="Error rate change percentage (positive = more errors)",
    )

    throughput_trend: float = Field(
        description="Throughput change percentage (positive = higher throughput)",
    )

    # Resource utilization trends
    cpu_usage_trend: float | None = Field(
        default=None,
        description="CPU usage change percentage",
    )

    memory_usage_trend: float | None = Field(
        default=None,
        description="Memory usage change percentage",
    )

    connection_count_trend: float = Field(
        description="Connection count change percentage",
    )

    # Predictive indicators
    projected_issues_count: int = Field(
        ge=0,
        description="Number of potential issues identified",
    )

    capacity_warning_threshold_hours: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated hours until capacity warning threshold",
    )

    # Data quality indicators
    data_points_analyzed: int = Field(
        ge=1,
        description="Number of data points used in analysis",
    )

    missing_data_periods: int = Field(
        ge=0,
        description="Number of periods with missing data",
    )

    # Historical context
    historical_data: list[ModelTrendDataPoint] | None = Field(
        default=None,
        max_items=1000,
        description="Historical data points used for trend analysis",
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
