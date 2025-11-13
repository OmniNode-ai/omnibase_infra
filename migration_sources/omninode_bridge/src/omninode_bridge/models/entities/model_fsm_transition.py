"""FSM transition entity model.

Maps to the fsm_transitions database table for tracking FSM state transitions.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelFSMTransition(BaseModel):
    """
    FSM transition entity model (maps to fsm_transitions table).

    Tracks finite state machine state transitions for workflows and aggregations,
    providing complete transition history and debugging capabilities.

    Attributes:
        id: Auto-generated primary key (UUID)
        entity_id: ID of the entity (workflow_id, bridge_id, etc.)
        entity_type: Type of entity (workflow, bridge_state, etc.)
        from_state: Previous state (None for initial state)
        to_state: New state after transition
        transition_event: Event that triggered the transition
        transition_data: Additional transition context as JSONB
        created_at: Record creation timestamp (auto-managed)

    Example:
        ```python
        from uuid import uuid4

        transition = ModelFSMTransition(
            entity_id=uuid4(),
            entity_type="workflow",
            from_state="PENDING",
            to_state="PROCESSING",
            transition_event="START_WORKFLOW",
            transition_data={"trigger": "user_request"}
        )
        ```
    """

    # Auto-generated primary key
    id: Optional[UUID] = Field(
        default=None, description="Auto-generated UUID primary key"
    )

    # Required fields
    entity_id: UUID = Field(..., description="ID of the entity undergoing transition")
    entity_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Type of entity (workflow, bridge_state, etc.)",
        examples=["workflow", "bridge_state", "aggregation"],
    )
    to_state: str = Field(
        ..., min_length=1, max_length=50, description="New state after transition"
    )
    transition_event: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Event that triggered the transition",
        examples=["START_WORKFLOW", "COMPLETE_STEP", "ERROR_OCCURRED"],
    )

    # Optional fields
    from_state: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Previous state (None for initial state)",
    )
    transition_data: dict[str, Any] = Field(
        default_factory=dict, description="Additional transition context as JSONB"
    )

    # Auto-managed timestamp
    created_at: Optional[datetime] = Field(
        default=None, description="Record creation timestamp (auto-managed by DB)"
    )

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate entity_type is one of the known types."""
        allowed_types = {
            "workflow",
            "workflow_execution",
            "bridge_state",
            "aggregation",
            "node",
        }
        if v.lower() not in allowed_types:
            # Allow other types but warn in logs
            pass
        return v.lower()

    @field_validator("to_state", "from_state")
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        """Normalize state values to uppercase."""
        if v is None:
            return None
        return v.upper()

    @field_validator("transition_event")
    @classmethod
    def validate_transition_event(cls, v: str) -> str:
        """Normalize transition_event to uppercase."""
        return v.upper()

    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode for DB row mapping
        json_schema_extra={
            "example": {
                "entity_id": "550e8400-e29b-41d4-a716-446655440000",
                "entity_type": "workflow",
                "from_state": "PENDING",
                "to_state": "PROCESSING",
                "transition_event": "START_WORKFLOW",
                "transition_data": {
                    "trigger": "user_request",
                    "user_id": "user-123",
                    "timestamp": "2025-10-08T12:00:00Z",
                },
            }
        },
    )
