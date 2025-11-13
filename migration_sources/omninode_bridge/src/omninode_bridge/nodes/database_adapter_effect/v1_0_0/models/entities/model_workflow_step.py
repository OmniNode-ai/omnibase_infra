#!/usr/bin/env python3
"""
ModelWorkflowStep - Workflow Step Entity.

Strongly-typed Pydantic model representing workflow_steps table.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelWorkflowStep(BaseModel):
    """Workflow step entity (maps to workflow_steps table)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Database-generated fields
    id: Optional[int] = Field(default=None, ge=1)

    # Workflow association
    workflow_id: UUID = Field(..., description="UUID of parent workflow execution")

    # Step identification
    step_name: str = Field(..., min_length=1, max_length=100)
    step_order: int = Field(..., ge=1)

    # Step status
    status: str = Field(..., min_length=1, max_length=50)

    # Performance metrics
    execution_time_ms: Optional[int] = Field(default=None, ge=0)

    # Step data (JSONB)
    step_data: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra={"db_type": "jsonb"}
    )

    # Error tracking
    error_message: Optional[str] = Field(default=None, max_length=2000)

    # Database timestamps
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelWorkflowStep":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
