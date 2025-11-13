#!/usr/bin/env python3
"""
ModelFSMTransition - FSM Transition Entity.

Strongly-typed Pydantic model representing fsm_transitions table.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    validate_jsonb_fields,
)


class ModelFSMTransition(BaseModel):
    """FSM state transition entity (maps to fsm_transitions table)."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Database-generated fields
    id: Optional[int] = Field(default=None, ge=1)

    # Entity identity
    entity_id: UUID = Field(..., description="UUID of entity undergoing transition")
    entity_type: str = Field(..., min_length=1, max_length=100)

    # State transition
    from_state: Optional[str] = Field(default=None, max_length=50)
    to_state: str = Field(..., min_length=1, max_length=50)

    # Transition event
    transition_event: str = Field(..., min_length=1, max_length=100)

    # Transition data (JSONB)
    transition_data: dict[str, Any] = Field(
        default_factory=dict, json_schema_extra={"db_type": "jsonb"}
    )

    # Database timestamps
    created_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelFSMTransition":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)
