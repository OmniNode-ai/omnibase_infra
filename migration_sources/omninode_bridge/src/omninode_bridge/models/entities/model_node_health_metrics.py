"""Node health metrics entity model.

Maps to the node_health_metrics database table for tracking node performance metrics.
"""

from datetime import datetime
from typing import Any, ClassVar, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ModelNodeHealthMetrics(BaseModel):
    """
    Node health metrics entity model (maps to node_health_metrics table).

    Tracks performance and health metrics for individual nodes over time,
    enabling trend analysis and alerting.

    Attributes:
        id: Auto-generated primary key (UUID)
        node_id: Reference to node in node_registrations table
        metric_type: Type of metric being tracked
        metric_value: Numeric value of the metric
        timestamp: Timestamp when metric was recorded
        metadata: Additional metric metadata as JSONB
        created_at: Record creation timestamp (auto-managed)

    Example:
        ```python
        from datetime import datetime, timezone

        metric = ModelNodeHealthMetrics(
            node_id="orchestrator-v1-instance-1",
            metric_type="cpu_usage_percent",
            metric_value=45.2,
            timestamp=datetime.now(timezone.utc),
            metadata={"cores": 4, "frequency_mhz": 2400}
        )
        ```
    """

    # Auto-generated primary key
    id: Optional[UUID] = Field(
        default=None, description="Auto-generated UUID primary key"
    )

    # Required fields
    node_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Reference to node in node_registrations table",
        examples=["orchestrator-v1-instance-1", "reducer-v1-instance-2"],
    )
    metric_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of metric being tracked",
        examples=[
            "cpu_usage_percent",
            "memory_usage_mb",
            "throughput_ops_per_sec",
            "latency_p95_ms",
            "error_rate_percent",
            "connection_pool_usage",
        ],
    )
    metric_value: float = Field(..., description="Numeric value of the metric")
    timestamp: datetime = Field(..., description="Timestamp when metric was recorded")

    # Optional fields
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metric metadata and context as JSONB",
    )

    # Auto-managed timestamp
    created_at: Optional[datetime] = Field(
        default=None, description="Record creation timestamp (auto-managed by DB)"
    )

    @field_validator("metric_type")
    @classmethod
    def validate_metric_type(cls, v: str) -> str:
        """Normalize metric_type to lowercase with underscores."""
        return v.lower().replace("-", "_").replace(" ", "_")

    @field_validator("metric_value")
    @classmethod
    def validate_metric_value(cls, v: float) -> float:
        """Validate metric_value is a valid number."""
        if not isinstance(v, int | float):
            raise ValueError(f"metric_value must be numeric, got: {type(v)}")
        # Allow negative values for certain metrics (e.g., temperature delta)
        return float(v)

    class Config:
        """Pydantic configuration."""

        from_attributes = True  # Enable ORM mode for DB row mapping
        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "node_id": "orchestrator-v1-instance-1",
                "metric_type": "cpu_usage_percent",
                "metric_value": 45.2,
                "timestamp": "2025-10-08T12:00:00Z",
                "metadata": {
                    "cores": 4,
                    "frequency_mhz": 2400,
                    "load_average": [1.5, 1.3, 1.2],
                },
            }
        }
