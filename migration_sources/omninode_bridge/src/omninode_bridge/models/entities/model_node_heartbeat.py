"""Node heartbeat entity model.

Maps to the node_registrations database table for service discovery and health tracking.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelNodeHeartbeat(BaseModel):
    """
    Node heartbeat entity model (maps to node_registrations table).

    Tracks node registration, health status, capabilities, and heartbeat timing
    for distributed node coordination and service discovery.

    Attributes:
        node_id: Primary key, unique node identifier (format: node_type-version-instance_id)
        node_type: Type of node (orchestrator, reducer, registry, etc.)
        node_version: Node version (e.g., v1.0.0)
        capabilities: Node capabilities and features as JSONB
        endpoints: Node endpoints for communication as JSONB
        metadata: Additional node metadata as JSONB
        health_status: Current health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
        last_heartbeat: Timestamp of last heartbeat signal
        registered_at: Timestamp when node was first registered (auto-managed)
        updated_at: Last update timestamp (auto-managed)

    Example:
        ```python
        from datetime import datetime, timezone

        node = ModelNodeHeartbeat(
            node_id="orchestrator-v1-instance-1",
            node_type="orchestrator",
            node_version="v1.0.0",
            capabilities={"max_concurrent_workflows": 100},
            endpoints={"health": "http://localhost:8080/health"},
            health_status="HEALTHY",
            last_heartbeat=datetime.now(timezone.utc)
        )
        ```
    """

    # Primary key (NOT auto-generated, provided by application)
    node_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique node identifier (PRIMARY KEY)",
        examples=["orchestrator-v1-instance-1", "reducer-v1-instance-2"],
    )

    # Required fields
    node_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of node (orchestrator, reducer, registry, etc.)",
        examples=["orchestrator", "reducer", "registry", "database_adapter"],
    )
    node_version: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Node version (e.g., v1.0.0)",
        examples=["v1.0.0", "v2.1.3"],
    )
    health_status: str = Field(
        default="UNKNOWN",
        min_length=1,
        max_length=50,
        description="Current health status",
        examples=["HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"],
    )

    # Optional fields
    capabilities: dict[str, Any] = Field(
        default_factory=dict, description="Node capabilities and features as JSONB"
    )
    endpoints: dict[str, Any] = Field(
        default_factory=dict, description="Node endpoints for communication as JSONB"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional node metadata as JSONB"
    )
    last_heartbeat: Optional[datetime] = Field(
        default=None, description="Timestamp of last heartbeat signal"
    )

    # Auto-managed timestamps
    registered_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when node was first registered (auto-managed by DB)",
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp (auto-managed by DB)"
    )

    @field_validator("health_status")
    @classmethod
    def validate_health_status(cls, v: str) -> str:
        """Validate and normalize health_status."""
        allowed_statuses = {"HEALTHY", "DEGRADED", "UNHEALTHY", "UNKNOWN"}
        v_upper = v.upper()
        if v_upper not in allowed_statuses:
            raise ValueError(
                f"health_status must be one of {allowed_statuses}, got: {v}"
            )
        return v_upper

    @field_validator("node_type")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        """Normalize node_type to lowercase."""
        return v.lower()

    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode for DB row mapping
        json_schema_extra={
            "example": {
                "node_id": "orchestrator-v1-instance-1",
                "node_type": "orchestrator",
                "node_version": "v1.0.0",
                "capabilities": {
                    "max_concurrent_workflows": 100,
                    "supported_operations": ["metadata_stamping", "intelligence"],
                },
                "endpoints": {
                    "health": "http://localhost:8080/health",
                    "metrics": "http://localhost:8080/metrics",
                },
                "metadata": {"region": "us-west-2", "environment": "production"},
                "health_status": "HEALTHY",
                "last_heartbeat": "2025-10-08T12:00:00Z",
            }
        },
    )
