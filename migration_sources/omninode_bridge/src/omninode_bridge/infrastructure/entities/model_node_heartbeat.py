#!/usr/bin/env python3
"""
ModelNodeHeartbeat - Node Heartbeat Entity.

Direct Pydantic representation of node_registrations database table.
Maps 1:1 with database schema for type-safe database operations.

Note: This entity uses the node_registrations table which tracks
both registration and heartbeat data for service discovery.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelNodeHeartbeat
- String node identifier
- Health status tracking
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelNodeHeartbeat(BaseModel):
    """
    Entity model for node_registrations table.

    This model represents node registration and heartbeat records in the database.
    It maps directly to the node_registrations table schema.

    Database Table: node_registrations
    Primary Key: node_id (VARCHAR, user-defined)
    Indexes: health_status, node_type, last_heartbeat

    Example:
        >>> from datetime import datetime, timezone
        >>> heartbeat = ModelNodeHeartbeat(
        ...     node_id="orchestrator-001",
        ...     node_type="orchestrator",
        ...     node_version="1.0.0",
        ...     health_status="HEALTHY",
        ...     last_heartbeat=datetime.now(timezone.utc)
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Primary key (user-defined string ID)
    node_id: str = Field(
        ...,
        description="Unique identifier for node instance",
        min_length=1,
        max_length=255,
    )

    # Node classification
    node_type: str = Field(
        ...,
        description="Type of node (orchestrator, reducer, registry, etc)",
        min_length=1,
        max_length=100,
    )

    node_version: str = Field(
        ...,
        description="Node version for compatibility tracking",
        min_length=1,
        max_length=50,
    )

    # Node capabilities
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Node capabilities and feature flags",
        json_schema_extra={"db_type": "jsonb"},
    )

    endpoints: dict[str, Any] = Field(
        default_factory=dict,
        description="Network endpoints for node communication",
        json_schema_extra={"db_type": "jsonb"},
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata",
        json_schema_extra={"db_type": "jsonb"},
    )

    # Health tracking
    health_status: str = Field(
        default="UNKNOWN",
        description="Current health status (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)",
        min_length=1,
        max_length=50,
    )

    last_heartbeat: Optional[datetime] = Field(
        default=None,
        description="Timestamp of most recent heartbeat signal",
    )

    # Auto-managed timestamps
    registered_at: Optional[datetime] = Field(
        default=None,
        description="Node registration timestamp (auto-managed by database)",
    )

    updated_at: Optional[datetime] = Field(
        default=None,
        description="Record update timestamp (auto-managed by database)",
    )

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelNodeHeartbeat":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
