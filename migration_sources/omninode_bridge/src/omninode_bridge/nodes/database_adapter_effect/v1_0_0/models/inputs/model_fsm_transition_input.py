#!/usr/bin/env python3
"""
ModelFSMTransitionInput - FSM State Transition Tracking Input.

Input model for persisting FSM state transitions in PostgreSQL.
Provides comprehensive audit trail of state machine transitions
across all bridge nodes.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelFSMTransitionInput
- UUID entity tracking
- Entity type polymorphism
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelFSMTransitionInput(BaseModel):
    """
    Input model for FSM state transition database operations.

    This model tracks all state transitions in finite state machines
    across the bridge node ecosystem. Each transition is recorded as
    an immutable audit record, enabling state history reconstruction
    and debugging.

    Database Table: fsm_transitions
    Primary Key: id (UUID, auto-generated)
    Indexes: entity_id + entity_type, created_at

    Entity Types:
    - "workflow": Workflow execution FSM
    - "bridge_reducer": Bridge reducer lifecycle FSM
    - "node_registry": Node registration FSM
    - "orchestrator": Orchestrator coordination FSM

    Common Transition Events:
    - "workflow_started": Workflow begins
    - "step_completed": Step finishes
    - "workflow_completed": Workflow succeeds
    - "workflow_failed": Workflow errors
    - "aggregation_started": Reducer begins aggregation
    - "aggregation_completed": Reducer finishes aggregation

    Event Sources:
    - STATE_TRANSITION: Published by all bridge nodes during FSM transitions

    Example (Workflow State Transition):
        >>> from uuid import uuid4
        >>> from datetime import datetime
        >>> transition = ModelFSMTransitionInput(
        ...     entity_id=uuid4(),
        ...     entity_type="workflow",
        ...     from_state="PROCESSING",
        ...     to_state="COMPLETED",
        ...     transition_event="workflow_completed",
        ...     transition_data={
        ...         "execution_time_ms": 1234,
        ...         "steps_completed": 5,
        ...         "items_processed": 10
        ...     }
        ... )

    Example (Initial State - No Previous State):
        >>> initial_transition = ModelFSMTransitionInput(
        ...     entity_id=uuid4(),
        ...     entity_type="workflow",
        ...     from_state=None,  # No previous state
        ...     to_state="PENDING",
        ...     transition_event="workflow_created",
        ...     transition_data={"namespace": "production"}
        ... )

    Example (Bridge Reducer Transition):
        >>> reducer_transition = ModelFSMTransitionInput(
        ...     entity_id=uuid4(),
        ...     entity_type="bridge_reducer",
        ...     from_state="idle",
        ...     to_state="aggregating",
        ...     transition_event="aggregation_started",
        ...     transition_data={
        ...         "window_size_ms": 5000,
        ...         "items_in_window": 100,
        ...         "namespace": "production"
        ...     }
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === Entity Identity ===
    entity_id: UUID = Field(
        ...,
        description="""
        UUID of the entity undergoing state transition.

        This could be:
        - Workflow correlation_id for workflow FSM
        - Bridge instance ID for reducer FSM
        - Node ID for registry FSM
        - Orchestrator instance ID for orchestrator FSM
        """,
    )

    entity_type: str = Field(
        ...,
        description="""
        Type of entity undergoing state transition.

        Common types:
        - "workflow": Workflow execution
        - "bridge_reducer": Bridge reducer instance
        - "node_registry": Node registration
        - "orchestrator": Orchestrator instance

        Used to distinguish between different FSM types
        when querying transition history.
        """,
        min_length=1,
        max_length=100,
    )

    # === State Transition ===
    from_state: str | None = Field(
        default=None,
        description="""
        Previous state before transition.

        Null for initial state transitions (entity creation).
        Otherwise, must be a valid state for the entity type.

        Example workflow states: PENDING, PROCESSING, COMPLETED, FAILED
        Example reducer states: idle, active, aggregating, persisting
        """,
        max_length=50,
    )

    to_state: str = Field(
        ...,
        description="""
        New state after transition.

        Must be a valid target state for the entity type.
        State machines define allowed transitions (from_state → to_state).

        Example workflow transitions:
        - PENDING → PROCESSING
        - PROCESSING → COMPLETED
        - PROCESSING → FAILED

        Example reducer transitions:
        - idle → active
        - active → aggregating
        - aggregating → persisting
        - persisting → idle
        """,
        min_length=1,
        max_length=50,
    )

    # === Transition Event ===
    transition_event: str = Field(
        ...,
        description="""
        Event that triggered this state transition.

        Events represent the actions or occurrences that caused
        the FSM to transition from one state to another.

        Common workflow events:
        - "workflow_created": Initial workflow creation
        - "workflow_started": Execution begins
        - "step_completed": Step finishes
        - "workflow_completed": Execution succeeds
        - "workflow_failed": Execution fails

        Common reducer events:
        - "items_received": Input items arrive
        - "window_timeout": Aggregation window closes
        - "aggregation_completed": Aggregation finishes
        - "persistence_completed": State saved
        """,
        min_length=1,
        max_length=100,
    )

    # === Transition Data ===
    transition_data: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Additional context and metadata for the transition.

        Example (Workflow Completion):
        {
            "execution_time_ms": 1234,
            "steps_completed": 5,
            "items_processed": 10,
            "final_state_data": {...}
        }

        Example (Workflow Failure):
        {
            "error_message": "Service timeout",
            "failed_step": "stamp_content",
            "retry_count": 3
        }

        Example (Reducer Aggregation):
        {
            "items_aggregated": 100,
            "window_duration_ms": 5000,
            "namespace": "production",
            "file_types": ["jpeg", "pdf"]
        }
        """,
    )

    # === Temporal Tracking ===
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="""
        Timestamp when this transition occurred.

        Automatically set to current UTC time.
        Used for transition history timeline and analytics.
        """,
    )
