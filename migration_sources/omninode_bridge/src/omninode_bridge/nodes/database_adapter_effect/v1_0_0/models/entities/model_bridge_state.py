#!/usr/bin/env python3
"""
ModelBridgeState - Bridge State Entity.

Strongly-typed Pydantic model representing bridge_states table.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)
from omninode_bridge.schemas import BRIDGE_STATE_METADATA_SCHEMA
from omninode_bridge.utils.metadata_validator import validate_metadata


class ModelBridgeState(BaseModel):
    """Bridge aggregation state entity (maps to bridge_states table)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Database-generated fields
    id: Optional[int] = Field(default=None, ge=1)

    # Bridge identity
    bridge_id: UUID = Field(..., description="Unique identifier for bridge instance")

    # Multi-tenant isolation
    namespace: str = Field(..., min_length=1, max_length=255)

    # Cumulative statistics
    total_workflows_processed: int = Field(..., ge=0)
    total_items_aggregated: int = Field(..., ge=0)

    # Aggregation metadata (JSONB)
    aggregation_metadata: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra={"db_type": "jsonb"}
    )

    # FSM state tracking
    current_fsm_state: str = Field(..., min_length=1, max_length=50)

    # Temporal tracking
    last_aggregation_timestamp: Optional[datetime] = Field(default=None)

    # Database timestamps
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    @field_validator("aggregation_metadata")
    @classmethod
    def validate_metadata_field(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate aggregation_metadata against JSON schema."""
        if v:
            validate_metadata(v, BRIDGE_STATE_METADATA_SCHEMA)
        return v

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelBridgeState":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
