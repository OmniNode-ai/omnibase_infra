#!/usr/bin/env python3
"""
ModelWorkflowProjection - Workflow Projection Entity.

Read-optimized projection of workflow state for fast queries.
Maps 1:1 with workflow_projection database table.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelWorkflowProjection
- Eventual consistency support with version gating
- Multi-tenant namespace isolation
- Comprehensive field validation with Pydantic v2

Pure Reducer Refactor - Wave 1, Workstream 1B
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelWorkflowProjection(BaseModel):
    """
    Entity model for workflow_projection table.

    This model represents read-optimized workflow state projections.
    It provides fast queries with eventual consistency guarantees.

    Database Table: workflow_projection
    Primary Key: workflow_key (TEXT)
    Indexes: namespace, tag, namespace+tag, version, indices (GIN)

    Consistency Model:
        - Eventually consistent with canonical state
        - Version gating prevents reading stale data
        - Fallback to canonical store if projection lags

    Example:
        >>> projection = ModelWorkflowProjection(
        ...     workflow_key="wf_123",
        ...     version=5,
        ...     tag="PROCESSING",
        ...     namespace="production",
        ...     last_action="StampContent"
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
    workflow_key: str = Field(
        ...,
        description="Unique workflow identifier (matches canonical store)",
        min_length=1,
        max_length=255,
    )

    # Version tracking (eventual consistency)
    version: int = Field(
        ...,
        description="Version number for eventual consistency gating",
        ge=1,
    )

    # FSM state tag
    tag: str = Field(
        ...,
        description="Workflow FSM state (PENDING, PROCESSING, COMPLETED, FAILED)",
        min_length=1,
        max_length=50,
    )

    # Last action applied
    last_action: Optional[str] = Field(
        default=None,
        description="Last action type applied (for debugging and tracing)",
        max_length=100,
    )

    # Multi-tenant isolation
    namespace: str = Field(
        ...,
        description="Namespace for multi-tenant isolation",
        min_length=1,
        max_length=255,
    )

    # Temporal tracking
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Last update timestamp (auto-managed by database)",
    )

    # Custom query indexes (JSONB)
    indices: Optional[dict[str, Any]] = Field(
        default=None,
        description="Custom query indexes for fast filtering (JSONB)",
        json_schema_extra={"db_type": "jsonb"},
    )

    # Additional metadata (JSONB)
    extras: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional projection-specific metadata (JSONB)",
        json_schema_extra={"db_type": "jsonb"},
    )

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelWorkflowProjection":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
