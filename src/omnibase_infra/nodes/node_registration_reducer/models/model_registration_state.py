# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration State Model for Pure Reducer Pattern.  # ai-slop-ok: pre-existing docstring opener

This module provides ModelRegistrationState, an immutable state model for the
registration reducer workflow. The state follows the pure reducer pattern
where state is passed in and returned from reduce(), with no internal mutation.

Architecture:
    ModelRegistrationState is designed for use with the canonical RegistrationReducer
    pattern. The state is:

    - Immutable (frozen=True): State transitions create new instances
    - Minimal: Only tracks essential workflow state
    - Type-safe: All fields have strict type annotations

    State transitions are performed via `with_*` methods that return new
    instances, ensuring the reducer remains pure and deterministic.

States:
    - idle: Waiting for introspection events
    - pending: Registration workflow started
    - partial: N/A (kept for backwards compat; resolves to pending now)
    - complete: Backend confirmed
    - failed: Validation or registration failed

State Management:
    This section documents the state lifecycle including immutability guarantees,
    transition methods, and integration with the persistence layer.

    **IMMUTABILITY GUARANTEES:**

    This model enforces strict immutability via Pydantic's frozen=True:

    - All field assignments after construction raise TypeError
    - State transitions return NEW instances; original is unchanged
    - This enables safe sharing across threads/async contexts
    - The reducer can safely compare old_state vs new_state

    Example of immutability behavior::

        state1 = ModelRegistrationState(status=EnumRegistrationStatus.IDLE)
        state2 = state1.with_pending_registration(node_id, event_id)

        # state1 is unchanged (immutable)
        assert state1.status == EnumRegistrationStatus.IDLE
        assert state2.status == EnumRegistrationStatus.PENDING

        # Attempting to mutate raises TypeError
        state1.status = EnumRegistrationStatus.PENDING  # Raises TypeError

    **STATE TRANSITION METHODS:**

    All state transitions are performed via ``with_*`` methods:

    - ``with_pending_registration(node_id, event_id)``: idle -> pending
    - ``with_postgres_confirmed(event_id)``: pending -> complete
    - ``with_failure(reason, event_id)``: any -> failed
    - ``with_reset(event_id)``: failed -> idle (recovery transition)

    Each method:

    1. Creates a NEW ModelRegistrationState instance
    2. Copies relevant fields from self
    3. Updates fields per transition logic
    4. Returns the new instance (self is unchanged)

    **INTEGRATION WITH PERSISTENCE LAYER:**

    This model is persisted to PostgreSQL by the Projector component:

    1. **Reducer Returns State**: After reduce() or reduce_confirmation(),
       the RegistrationReducer returns ModelReducerOutput containing the
       new state in the ``result`` field.

    2. **Runtime Extracts State**: The Runtime extracts the state from
       ModelReducerOutput.result for persistence.

    3. **Projector Persists State**: The Projector writes the state to
       PostgreSQL synchronously before any Kafka publishing.

    4. **Serialization**: The Projector uses Pydantic's ``model_dump(mode="json")``
       to serialize state for PostgreSQL storage.

    **IDEMPOTENCY VIA last_processed_event_id:**

    The ``last_processed_event_id`` field enables idempotent event processing:

    - Each event has a unique ID (correlation_id or generated UUID)
    - Before processing, the reducer checks ``state.is_duplicate_event(event_id)``
    - If True, the event was already processed; reducer returns current state
    - This enables safe replay after crashes or redelivery

Related:
    - RegistrationReducer: Pure reducer that uses this state model
    - OMN-889: Infrastructure MVP
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumRegistrationStatus

# Type alias for failure reason literals
FailureReason = Literal[
    "validation_failed",
    "consul_failed",  # kept for existing DB records; no longer emitted by reducer
    "postgres_failed",
    "both_failed",
    "invalid_reset_state",
]


class ModelRegistrationState(BaseModel):
    """State model for the registration reducer workflow.

    Immutable state passed to and returned from reduce().
    Follows pure reducer pattern - no internal state mutation.

    The state tracks the current workflow status and confirmation state
    for the PostgreSQL backend. State transitions are performed via
    ``with_*`` methods that return new immutable instances.

    Note on consul_confirmed:
        This field is retained for backwards compatibility with existing
        PostgreSQL projection rows. It is no longer set to True by any
        reducer code (OMN-3540). New rows will always have consul_confirmed=False.

    Persistence Integration:
        This model is designed for persistence to PostgreSQL via the Projector:

        - **Stored**: By Runtime calling Projector.persist() after reduce() returns
        - **Retrieved**: By Orchestrator via ProtocolProjectionReader before reduce()
        - **Idempotency**: ``last_processed_event_id`` enables duplicate detection

        The reducer does NOT persist state directly - it returns the new state
        in ModelReducerOutput.result. The Runtime handles persistence.

    Immutability:
        This model uses frozen=True to enforce strict immutability:

        - All fields are immutable after construction
        - Transition methods (with_*) return NEW instances
        - Original state is never modified
        - Safe for concurrent access and comparison

    Attributes:
        status: Current workflow status (idle, pending, partial, complete, failed).
        node_id: UUID of the node being registered, if any.
        consul_confirmed: Deprecated. Always False for new rows (OMN-3540).
        postgres_confirmed: Whether PostgreSQL registration is confirmed.
        last_processed_event_id: UUID of last processed event for idempotency.
        failure_reason: Reason for failure, if status is "failed".

    Example:
        >>> from uuid import uuid4
        >>> state = ModelRegistrationState()  # Initial idle state
        >>> state.status
        'idle'
        >>> node_id, event_id = uuid4(), uuid4()
        >>> state = state.with_pending_registration(node_id, event_id)
        >>> state.status
        'pending'
        >>> state = state.with_postgres_confirmed(uuid4())
        >>> state.status
        'complete'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumRegistrationStatus = Field(
        default=EnumRegistrationStatus.IDLE,
        description="Current workflow status",
    )
    node_id: UUID | None = Field(
        default=None,
        description="Node being registered",
    )
    consul_confirmed: bool = Field(
        default=False,
        description="Deprecated: Consul was removed (OMN-3540). Always False for new rows.",
    )
    postgres_confirmed: bool = Field(
        default=False,
        description="Whether PostgreSQL registration is confirmed",
    )
    last_processed_event_id: UUID | None = Field(
        default=None,
        description="Last processed event ID for idempotency",
    )
    failure_reason: FailureReason | None = Field(
        default=None,
        description="Reason for failure, if status is failed",
    )

    def with_pending_registration(
        self, node_id: UUID, event_id: UUID
    ) -> ModelRegistrationState:
        """Transition to pending state for a new registration.

        Creates a new state instance with status="pending" and the given
        node_id. Resets confirmation flags and clears any failure reason.

        Args:
            node_id: UUID of the node being registered.
            event_id: UUID of the event triggering this transition.

        Returns:
            New ModelRegistrationState with pending status.
        """
        return ModelRegistrationState(
            status=EnumRegistrationStatus.PENDING,
            node_id=node_id,
            consul_confirmed=False,
            postgres_confirmed=False,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def with_postgres_confirmed(self, event_id: UUID) -> ModelRegistrationState:
        """Transition state after PostgreSQL registration is confirmed.

        Status becomes "complete" when PostgreSQL confirms.

        Args:
            event_id: UUID of the event confirming PostgreSQL registration.

        Returns:
            New ModelRegistrationState with postgres_confirmed=True and status=complete.
        """
        return ModelRegistrationState(
            status=EnumRegistrationStatus.COMPLETE,
            node_id=self.node_id,
            consul_confirmed=self.consul_confirmed,
            postgres_confirmed=True,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def with_failure(
        self, reason: FailureReason, event_id: UUID
    ) -> ModelRegistrationState:
        """Transition to failed state with a reason.

        Preserves current confirmation flags for diagnostic purposes.

        Args:
            reason: The failure reason (validation_failed, postgres_failed,
                both_failed, or invalid_reset_state).
            event_id: UUID of the event triggering the failure.

        Returns:
            New ModelRegistrationState with status="failed" and failure_reason set.
        """
        return ModelRegistrationState(
            status=EnumRegistrationStatus.FAILED,
            node_id=self.node_id,
            consul_confirmed=self.consul_confirmed,
            postgres_confirmed=self.postgres_confirmed,
            last_processed_event_id=event_id,
            failure_reason=reason,
        )

    def is_duplicate_event(self, event_id: UUID) -> bool:
        """Check if an event has already been processed.

        Used for idempotency to skip duplicate event processing.

        Args:
            event_id: UUID of the event to check.

        Returns:
            True if this event_id matches the last processed event.
        """
        return self.last_processed_event_id == event_id

    def with_reset(self, event_id: UUID) -> ModelRegistrationState:
        """Transition from failed state back to idle for retry.

        Allows recovery from failed states by resetting to idle. This enables
        the FSM to process new introspection events after a failure.

        This method can be called from any state but is primarily intended
        for recovery from the failed state. All confirmation flags are reset
        and the failure reason is cleared.

        State Diagram::

            +--------+   reset event   +------+
            | failed | --------------> | idle |
            +--------+                 +------+

        Args:
            event_id: UUID of the reset event triggering this transition.

        Returns:
            New ModelRegistrationState with status="idle" and all flags reset.

        Example:
            >>> from uuid import uuid4
            >>> from omnibase_infra.enums import EnumRegistrationStatus
            >>> state = ModelRegistrationState(
            ...     status=EnumRegistrationStatus.FAILED, failure_reason="postgres_failed"
            ... )
            >>> reset_state = state.with_reset(uuid4())
            >>> reset_state.status == EnumRegistrationStatus.IDLE
            True
            >>> reset_state.failure_reason is None
            True
        """
        return ModelRegistrationState(
            status=EnumRegistrationStatus.IDLE,
            node_id=None,
            consul_confirmed=False,
            postgres_confirmed=False,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def can_reset(self) -> bool:
        """Check if the current state allows reset to idle.

        Returns True if the state is in a terminal or error state that
        can be reset. This includes 'failed' and 'complete' states.

        Returns:
            True if reset is allowed from the current state.
        """
        return self.status in (
            EnumRegistrationStatus.FAILED,
            EnumRegistrationStatus.COMPLETE,
        )


__all__ = ["ModelRegistrationState"]
