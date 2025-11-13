#!/usr/bin/env python3
"""
ModelNodeHealthMetrics - Node Health Metrics Entity.

Virtual entity model for node health metrics.
Currently uses node_registrations table but may have dedicated table in future.

This entity model represents the health metrics aspect of node monitoring,
focusing on performance and availability metrics.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelNodeHealthMetrics
- String node identifier
- Performance metrics tracking
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelNodeHealthMetrics(BaseModel):
    """
    Entity model for node health metrics.

    This is a virtual entity that currently maps to node_registrations table,
    but focuses on the health metrics aspect of node monitoring.

    In future versions, this may map to a dedicated node_health_metrics table
    for detailed performance tracking and time-series health data.

    Database Table: node_registrations (currently) / node_health_metrics (future)
    Primary Key: node_id (VARCHAR)
    Indexes: health_status, last_heartbeat

    Example:
        >>> from datetime import datetime, timezone
        >>> metrics = ModelNodeHealthMetrics(
        ...     node_id="orchestrator-001",
        ...     node_type="orchestrator",
        ...     health_status="HEALTHY",
        ...     performance_metrics={
        ...         "cpu_usage": 45.2,
        ...         "memory_usage_mb": 512,
        ...         "active_workflows": 10
        ...     },
        ...     last_health_check=datetime.now(timezone.utc)
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Node identity
    node_id: str = Field(
        ...,
        description="Unique identifier for node instance",
        min_length=1,
        max_length=255,
    )

    node_type: str = Field(
        ...,
        description="Type of node (orchestrator, reducer, registry, etc)",
        min_length=1,
        max_length=100,
    )

    # Health status
    health_status: str = Field(
        ...,
        description="Current health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)",
        min_length=1,
        max_length=50,
    )

    # Performance metrics
    performance_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Performance metrics for the node.

        Example:
        {
            "cpu_usage_percent": 45.2,
            "memory_usage_mb": 512,
            "active_workflows": 10,
            "operations_per_second": 100.5,
            "avg_response_time_ms": 5.2,
            "error_rate_percent": 0.1
        }
        """,
        json_schema_extra={"db_type": "jsonb"},
    )

    # Availability metrics
    uptime_seconds: Optional[int] = Field(
        default=None,
        description="Node uptime in seconds",
        ge=0,
    )

    # Temporal tracking
    last_health_check: Optional[datetime] = Field(
        default=None,
        description="Timestamp of most recent health check",
    )

    # Auto-managed timestamps
    created_at: Optional[datetime] = Field(
        default=None,
        description="Record creation timestamp (auto-managed by database)",
    )

    updated_at: Optional[datetime] = Field(
        default=None,
        description="Record update timestamp (auto-managed by database)",
    )

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelNodeHealthMetrics":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
