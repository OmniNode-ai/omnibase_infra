#!/usr/bin/env python3
"""
ModelBridgeState - Bridge State Entity.

Direct Pydantic representation of bridge_states database table.
Maps 1:1 with database schema for type-safe database operations.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelBridgeState
- UUID bridge identifier
- Aggregation state tracking
- Comprehensive field validation with Pydantic v2
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
    """
    Entity model for bridge_states table.

    This model represents bridge aggregation state records in the database.
    It maps directly to the bridge_states table schema.

    Database Table: bridge_states
    Primary Key: bridge_id (UUID)
    Indexes: namespace, current_fsm_state, last_aggregation_timestamp

    Example:
        >>> from uuid import uuid4
        >>> from datetime import datetime
        >>> bridge = ModelBridgeState(
        ...     bridge_id=uuid4(),
        ...     namespace="production",
        ...     total_workflows_processed=100,
        ...     total_items_aggregated=1000,
        ...     current_fsm_state="PROCESSING"
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Primary key
    bridge_id: UUID = Field(
        ...,
        description="Unique identifier for bridge instance",
    )

    # Multi-tenant isolation
    namespace: str = Field(
        ...,
        description="Namespace for multi-tenant isolation",
        min_length=1,
        max_length=255,
    )

    # Aggregation counters
    total_workflows_processed: int = Field(
        default=0,
        description="Cumulative count of workflows processed",
        ge=0,
    )

    total_items_aggregated: int = Field(
        default=0,
        description="Cumulative count of items aggregated",
        ge=0,
    )

    # Aggregation metadata
    aggregation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregation statistics and metadata",
        json_schema_extra={"db_type": "jsonb"},
    )

    # FSM state
    current_fsm_state: str = Field(
        ...,
        description="Current finite state machine state",
        min_length=1,
        max_length=50,
    )

    # Temporal tracking
    last_aggregation_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of most recent aggregation",
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
