#!/usr/bin/env python3
"""
ModelWorkflowState - Canonical Workflow State Entity.

Direct Pydantic representation of workflow_state database table.
Maps 1:1 with database schema for type-safe database operations.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelWorkflowState
- Version-based optimistic concurrency control
- JSONB state and provenance fields
- Comprehensive field validation with Pydantic v2

Pure Reducer Refactor:
- Single source of truth for workflow state
- Optimistic concurrency with version field
- Provenance tracking for all state changes
- Schema versioning for future migrations

Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md (Wave 1, Workstream 1A)
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omninode_bridge.infrastructure.validation.jsonb_validators import (
    JsonbField,
    validate_jsonb_fields,
)


class ModelWorkflowState(BaseModel):
    """
    Entity model for workflow_state table.

    This model represents the canonical workflow state record in the database.
    It serves as the single source of truth for all workflow state with
    version-based optimistic concurrency control.

    Database Table: workflow_state
    Primary Key: workflow_key (TEXT)
    Concurrency Control: version (BIGINT)
    Indexes: (workflow_key, version), updated_at DESC

    Optimistic Concurrency Pattern:
    1. Read state with current version
    2. Compute new state with pure reducer
    3. Try to commit with expected version
    4. If version mismatch, retry with backoff

    Example:
        >>> from datetime import datetime, timezone
        >>> state = ModelWorkflowState(
        ...     workflow_key="workflow-123",
        ...     version=1,
        ...     state={"items": [], "count": 0},
        ...     updated_at=datetime.now(timezone.utc),
        ...     schema_version=1,
        ...     provenance={
        ...         "effect_id": "effect-456",
        ...         "timestamp": "2025-10-21T12:00:00Z",
        ...         "action_id": "action-789"
        ...     }
        ... )
        >>> assert state.version >= 1
        >>> assert "items" in state.state
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Primary key: human-readable workflow identifier
    workflow_key: str = Field(
        ...,
        description="Human-readable workflow identifier (PRIMARY KEY)",
        min_length=1,
        max_length=255,
    )

    # Version number for optimistic concurrency control
    version: int = Field(
        ...,
        description="Version number for optimistic locking (incremented on each update)",
        ge=1,
    )

    # Current workflow state as JSONB
    state: dict[str, Any] = JsonbField(
        ...,
        description="Current workflow state as JSONB",
    )

    # Timestamp of last state update
    updated_at: datetime = Field(
        ...,
        description="Timestamp of last state update (auto-managed by database)",
    )

    # Schema version for future migrations
    schema_version: int = Field(
        default=1,
        description="Schema version for future migrations",
        ge=1,
    )

    # Provenance metadata
    provenance: dict[str, Any] = JsonbField(
        ...,
        description="Provenance metadata (effect_id, timestamp, action_id, etc.)",
    )

    @model_validator(mode="after")
    def _validate_jsonb_fields(self) -> "ModelWorkflowState":
        """Validate that all JSONB fields have proper annotations."""
        return validate_jsonb_fields(self)

    @model_validator(mode="after")
    def _validate_version_positive(self) -> "ModelWorkflowState":
        """Validate that version is positive (>= 1)."""
        if self.version < 1:
            raise ValueError(
                f"Version must be >= 1, got {self.version}. "
                "Version starts at 1 and is incremented on each state update."
            )
        return self

    @model_validator(mode="after")
    def _validate_state_not_empty(self) -> "ModelWorkflowState":
        """Validate that state is not an empty dict."""
        if not self.state:
            raise ValueError(
                "State cannot be empty. Workflow state must contain at least one field."
            )
        return self

    @model_validator(mode="after")
    def _validate_provenance_required_fields(self) -> "ModelWorkflowState":
        """Validate that provenance contains required fields."""
        required_fields = ["effect_id", "timestamp"]
        missing_fields = [
            field for field in required_fields if field not in self.provenance
        ]

        if missing_fields:
            raise ValueError(
                f"Provenance missing required fields: {missing_fields}. "
                f"Required fields: {required_fields}"
            )

        return self
