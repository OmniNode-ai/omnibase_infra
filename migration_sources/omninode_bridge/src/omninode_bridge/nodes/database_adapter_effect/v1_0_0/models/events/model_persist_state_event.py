"""
PersistState Event Model - Pure Reducer Refactor Wave 4A.

Event published by ReducerService to trigger state persistence in Store Effect Node.
This event represents the intent to persist workflow state with optimistic concurrency control.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelPersistStateEvent
- Strong typing with Pydantic models
- Event-driven architecture
- Comprehensive error handling

Pure Reducer Refactor - Wave 4, Workstream 4A
Reference: docs/planning/PURE_REDUCER_REFACTOR_PLAN.md
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelPersistStateEvent(BaseModel):
    """
    Event to trigger state persistence with optimistic concurrency control.

    Published by ReducerService after pure reducer generates new state.
    Consumed by Store Effect Node to persist state via CanonicalStoreService.

    Event Flow:
    1. ReducerService → Publishes PersistState event to Kafka
    2. Store Effect Node → Subscribes to event
    3. Store Effect Node → Calls canonical_store.try_commit()
    4. Store Effect Node → Publishes StateCommitted or StateConflict event

    Attributes:
        event_type: Event type identifier (always "PersistState")
        workflow_key: Human-readable workflow identifier
        expected_version: Expected current version (optimistic lock)
        state_prime: New state to commit (complete state snapshot)
        action_id: Action ID that triggered this state change
        provenance: Provenance metadata (effect_id, timestamp, etc.)
        correlation_id: Correlation ID for tracing
        timestamp: Event timestamp (UTC)
    """

    event_type: str = Field(
        default="PersistState",
        description="Event type identifier",
    )

    workflow_key: str = Field(
        ...,
        description="Workflow identifier",
        min_length=1,
    )

    expected_version: int = Field(
        ...,
        description="Expected current version for optimistic lock",
        ge=1,
    )

    state_prime: dict[str, Any] = Field(
        ...,
        description="New state to commit (complete state snapshot)",
    )

    action_id: UUID | None = Field(
        default=None,
        description="Action ID that triggered this state change",
    )

    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance metadata (effect_id, timestamp, etc.)",
    )

    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp (UTC)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "event_type": "PersistState",
                "workflow_key": "workflow-123",
                "expected_version": 1,
                "state_prime": {
                    "aggregations": {
                        "omninode.services.metadata": {
                            "total_stamps": 100,
                            "total_size_bytes": 1024000,
                        }
                    }
                },
                "action_id": "550e8400-e29b-41d4-a716-446655440000",
                "provenance": {
                    "effect_id": "effect-456",
                    "timestamp": "2025-10-21T12:00:00Z",
                    "action_id": "action-789",
                },
                "correlation_id": "650e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-10-21T12:00:00Z",
            }
        },
    )
