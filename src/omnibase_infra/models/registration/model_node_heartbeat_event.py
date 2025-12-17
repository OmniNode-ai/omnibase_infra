# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node heartbeat event model for periodic health broadcasts."""

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ModelNodeHeartbeatEvent(BaseModel):
    """Event model for periodic node heartbeat broadcasts.

    This model represents heartbeat events emitted by nodes to indicate
    they are alive and operational. Used by IntrospectionMixin to broadcast
    periodic health status to the event bus.

    Attributes:
        node_id: Unique identifier for the node instance.
        node_type: Type classification of the node (e.g., EFFECT, COMPUTE).
        uptime_seconds: Time in seconds since the node started.
        active_operations_count: Number of currently active operations.
        memory_usage_mb: Optional memory usage in megabytes.
        cpu_usage_percent: Optional CPU usage percentage (0-100).
        correlation_id: Optional correlation ID for tracing.
        timestamp: UTC timestamp when the heartbeat was generated.
    """

    node_id: str = Field(..., description="Node identifier")
    node_type: str = Field(..., description="Node type classification")

    # Health metrics
    uptime_seconds: float = Field(ge=0, description="Seconds since node startup")
    active_operations_count: int = Field(
        ge=0, default=0, description="Number of active operations"
    )

    # Resource usage (optional)
    memory_usage_mb: float | None = Field(
        default=None, description="Memory usage in megabytes"
    )
    cpu_usage_percent: float | None = Field(
        default=None, ge=0, le=100, description="CPU usage percentage"
    )

    # Metadata
    correlation_id: UUID | None = Field(
        default=None, description="Correlation ID for distributed tracing"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of heartbeat generation",
    )

    model_config = {
        "frozen": False,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "node_id": "node-postgres-adapter-001",
                    "node_type": "EFFECT",
                    "uptime_seconds": 3600.5,
                    "active_operations_count": 3,
                    "memory_usage_mb": 128.5,
                    "cpu_usage_percent": 12.3,
                    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                    "timestamp": "2025-01-15T10:30:00Z",
                }
            ]
        },
    }
