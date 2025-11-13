#!/usr/bin/env python3
"""
ModelStateCommittedEvent - State Committed Event Entity.

Event emitted when workflow state is successfully committed to canonical store.
Used by Projection Materializer to update read-optimized projections.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelStateCommittedEvent
- Event-driven architecture pattern
- Comprehensive field validation with Pydantic v2

Pure Reducer Refactor - Wave 2, Workstream 2C
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omninode_bridge.infrastructure.validation.jsonb_validators import JsonbField


class ModelStateCommittedEvent(BaseModel):
    """
    Entity model for StateCommitted events published to Kafka.

    This event is emitted by CanonicalStoreService when workflow state
    is successfully committed with optimistic concurrency control.
    ProjectionMaterializerService subscribes to these events to update
    read-optimized projections.

    Kafka Topic: dev.omninode_bridge.onex.evt.state-committed.v1
    Consumer Group: projection-materializer

    Event Flow:
    1. CanonicalStoreService commits state with version increment
    2. Publishes StateCommitted event to Kafka
    3. ProjectionMaterializer consumes event
    4. Updates projection + watermark atomically

    Example:
        >>> from datetime import datetime, timezone
        >>> from uuid import uuid4
        >>> event = ModelStateCommittedEvent(
        ...     event_id=uuid4(),
        ...     workflow_key="workflow-123",
        ...     version=2,
        ...     state={"items": ["a", "b"], "count": 2},
        ...     tag="PROCESSING",
        ...     namespace="production",
        ...     provenance={
        ...         "effect_id": str(uuid4()),
        ...         "timestamp": datetime.now(timezone.utc).isoformat(),
        ...         "action_id": str(uuid4())
        ...     },
        ...     committed_at=datetime.now(timezone.utc),
        ...     partition_id="kafka-partition-0",
        ...     offset=12345
        ... )
        >>> assert event.version >= 1
        >>> assert event.tag in ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
        populate_by_name=True,
    )

    # Event metadata
    event_id: UUID = Field(
        ...,
        description="Unique event identifier",
    )

    # Workflow identification
    workflow_key: str = Field(
        ...,
        description="Workflow identifier (matches canonical store)",
        min_length=1,
        max_length=255,
    )

    # Version tracking
    version: int = Field(
        ...,
        description="New version after commit (post-increment)",
        ge=1,
    )

    # Workflow state
    state: dict[str, Any] = JsonbField(
        ...,
        description="Committed workflow state (JSONB)",
    )

    # FSM state tag
    tag: str = Field(
        ...,
        description="Workflow FSM state tag (PENDING, PROCESSING, COMPLETED, FAILED)",
        min_length=1,
        max_length=50,
    )

    # Last action applied
    last_action: str | None = Field(
        default=None,
        description="Last action type that triggered this state change",
        max_length=100,
    )

    # Multi-tenant isolation
    namespace: str = Field(
        ...,
        description="Namespace for multi-tenant isolation",
        min_length=1,
        max_length=255,
    )

    # Provenance metadata
    provenance: dict[str, Any] = JsonbField(
        ...,
        description="Provenance metadata (effect_id, timestamp, action_id, etc.)",
    )

    # Temporal tracking
    committed_at: datetime = Field(
        ...,
        description="Timestamp when state was committed to canonical store",
    )

    # Kafka metadata (for watermark tracking)
    partition_id: str = Field(
        ...,
        description="Kafka partition ID for watermark tracking",
        min_length=1,
        max_length=255,
    )

    offset: int = Field(
        ...,
        description="Kafka offset for idempotence and watermark tracking",
        ge=0,
    )

    # Custom query indexes (for projection)
    indices: dict[str, Any] | None = Field(
        default=None,
        description="Custom query indexes for projection (JSONB)",
    )

    # Additional metadata
    extras: dict[str, Any] | None = Field(
        default=None,
        description="Additional event-specific metadata (JSONB)",
    )
