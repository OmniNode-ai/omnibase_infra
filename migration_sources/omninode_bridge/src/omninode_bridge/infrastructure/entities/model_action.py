"""
Action Model for Pure Reducer Refactor.

Represents an action to be processed by the reducer with
deduplication support and workflow tracking.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelAction
- Strong typing with Pydantic v2
- UUID-based action identification
- Workflow key correlation

Pure Reducer Refactor - Wave 3, Workstream 3B
Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md
"""

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field


class ModelAction(BaseModel):
    """
    Action model for reducer processing.

    Represents a single action to be applied to workflow state via
    the reducer. Actions are deduplicated by (workflow_key, action_id)
    and processed idempotently.

    Attributes:
        action_id: Unique identifier for this action (for deduplication)
        workflow_key: Workflow identifier this action applies to
        epoch: Action epoch (for ordering and versioning)
        lease_id: Lease identifier for action ownership
        payload: Action-specific data to be processed
        timestamp: When action was created (UTC)

    Example:
        >>> from uuid import uuid4
        >>> action = ModelAction(
        ...     action_id=uuid4(),
        ...     workflow_key="workflow-123",
        ...     epoch=1,
        ...     lease_id=uuid4(),
        ...     payload={"operation": "add_stamp", "data": {...}}
        ... )
        >>> assert action.workflow_key == "workflow-123"
    """

    action_id: UUID = Field(
        ...,
        description="Unique action identifier for deduplication",
    )
    workflow_key: str = Field(
        ...,
        description="Workflow identifier this action applies to",
        min_length=1,
    )
    epoch: int = Field(
        default=1,
        description="Action epoch for ordering",
        ge=1,
    )
    lease_id: UUID = Field(
        ...,
        description="Lease identifier for action ownership",
    )
    payload: dict = Field(
        default_factory=dict,
        description="Action-specific data to be processed by reducer",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Action creation timestamp (UTC)",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "action_id": "550e8400-e29b-41d4-a716-446655440000",
                "workflow_key": "workflow-123",
                "epoch": 1,
                "lease_id": "660e8400-e29b-41d4-a716-446655440000",
                "payload": {
                    "operation": "add_stamp",
                    "namespace": "omninode.services.metadata",
                    "data": {"file_hash": "abc123", "file_size": 1024},
                },
                "timestamp": "2025-10-21T12:00:00Z",
            }
        }
