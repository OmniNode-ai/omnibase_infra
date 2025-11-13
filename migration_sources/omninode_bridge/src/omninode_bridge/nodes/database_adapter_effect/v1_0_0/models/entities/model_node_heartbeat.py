#!/usr/bin/env python3
"""
ModelNodeHeartbeat - Node Heartbeat Entity.

Strongly-typed Pydantic model representing node_registrations table.
"""

from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelNodeHeartbeat(BaseModel):
    """Node heartbeat entity (maps to node_registrations table)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Database-generated fields (for consistency with other entities)
    id: Optional[int] = Field(default=None, ge=1)

    # Node identity (primary key in database)
    node_id: str = Field(..., min_length=1, max_length=255)

    # Health status
    health_status: str = Field(..., min_length=1, max_length=50)

    # Heartbeat metadata (JSONB)
    metadata: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra={"db_type": "jsonb"}
    )

    # Temporal tracking
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Database timestamps
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelNodeHeartbeat":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
