# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration State Model for Pure Reducer Pattern.

This module provides ModelRegistrationState, an immutable state model for the
dual registration reducer workflow. The state follows the pure reducer pattern
where state is passed in and returned from reduce(), with no internal mutation.

Architecture:
    ModelRegistrationState is designed for use with the canonical RegistrationReducer
    pattern defined in DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md. The state is:

    - Immutable (frozen=True): State transitions create new instances
    - Minimal: Only tracks essential workflow state
    - Type-safe: All fields have strict type annotations

    State transitions are performed via `with_*` methods that return new
    instances, ensuring the reducer remains pure and deterministic.

States:
    - idle: Waiting for introspection events
    - pending: Registration workflow started
    - partial: One backend confirmed, waiting for the other
    - complete: Both backends confirmed
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
    - ``with_consul_confirmed(event_id)``: pending -> partial, or partial -> complete
    - ``with_postgres_confirmed(event_id)``: pending -> partial, or partial -> complete
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

    PostgreSQL Storage::

        # Conceptual Projector implementation
        async def persist(self, state: ModelRegistrationState, offset: int) -> None:
            '''Persist state to PostgreSQL with idempotency.'''
            await self.db.execute(
                '''
                UPDATE node_registrations
                SET
                    status = $1,
                    consul_confirmed = $2,
                    postgres_confirmed = $3,
                    last_processed_event_id = $4,
                    failure_reason = $5,
                    last_event_offset = $6,
                    updated_at = NOW()
                WHERE
                    node_id = $7
                    AND (last_event_offset IS NULL OR $6 > last_event_offset)
                ''',
                state.status,
                state.consul_confirmed,
                state.postgres_confirmed,
                state.last_processed_event_id,
                state.failure_reason,
                offset,
                state.node_id,
            )

    **State Loading (Projection Reader)**::

        # Conceptual ProtocolProjectionReader implementation
        async def get_projection(
            self, entity_type: str, entity_id: UUID
        ) -> ModelRegistrationState | None:
            '''Load state from PostgreSQL projection.'''
            row = await self.db.fetchone(
                'SELECT * FROM node_registrations WHERE node_id = $1',
                entity_id,
            )
            if row is None:
                return None  # Orchestrator creates initial idle state
            return ModelRegistrationState(
                status=row['status'],
                node_id=row['node_id'],
                consul_confirmed=row['consul_confirmed'],
                postgres_confirmed=row['postgres_confirmed'],
                last_processed_event_id=row['last_processed_event_id'],
                failure_reason=row['failure_reason'],
            )

    **IDEMPOTENCY VIA last_processed_event_id:**

    The ``last_processed_event_id`` field enables idempotent event processing:

    - Each event has a unique ID (correlation_id or generated UUID)
    - Before processing, the reducer checks ``state.is_duplicate_event(event_id)``
    - If True, the event was already processed; reducer returns current state
    - This enables safe replay after crashes or redelivery

    Two levels of idempotency:

    1. **Reducer Level**: ``is_duplicate_event()`` checks ``last_processed_event_id``
    2. **Persistence Level**: Projector checks ``last_event_offset`` in SQL

    Both levels are required for full replay safety.

Related:
    - RegistrationReducer: Pure reducer that uses this state model
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
    - ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md: Tickets F0, F1, B3
    - OMN-889: Infrastructure MVP
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumRegistrationStatus

# Type alias for failure reason literals
FailureReason = Literal[
    "validation_failed",
    "consul_failed",
    "postgres_failed",
    "both_failed",
    "invalid_reset_state",
]


class ModelRegistrationState(BaseModel):
    """State model for the dual registration reducer workflow.

    Immutable state passed to and returned from reduce().
    Follows pure reducer pattern - no internal state mutation.

    The state tracks the current workflow status and confirmation state
    for both Consul and PostgreSQL backends. State transitions are
    performed via ``with_*`` methods that return new immutable instances.

    Persistence Integration:
        This model is designed for persistence to PostgreSQL via the Projector:

        - **Stored**: By Runtime calling Projector.persist() after reduce() returns
        - **Retrieved**: By Orchestrator via ProtocolProjectionReader before reduce()
        - **Idempotency**: ``last_processed_event_id`` enables duplicate detection

        The reducer does NOT persist state directly - it returns the new state
        in ModelReducerOutput.result. The Runtime handles persistence.

        See the module docstring "State Management" section for complete details
        on persistence integration, including PostgreSQL schema and example code.

    Immutability:
        This model uses frozen=True to enforce strict immutability:

        - All fields are immutable after construction
        - Transition methods (with_*) return NEW instances
        - Original state is never modified
        - Safe for concurrent access and comparison

    Attributes:
        status: Current workflow status (idle, pending, partial, complete, failed).
        node_id: UUID of the node being registered, if any.
        consul_confirmed: Whether Consul registration is confirmed.
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
        >>> state = state.with_consul_confirmed(uuid4())
        >>> state.status
        'partial'
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
        description="Whether Consul registration is confirmed",
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

    def with_consul_confirmed(self, event_id: UUID) -> ModelRegistrationState:
        """Transition state after Consul registration is confirmed.

        If PostgreSQL is already confirmed, status becomes "complete".
        Otherwise, status becomes "partial".

        Args:
            event_id: UUID of the event confirming Consul registration.

        Returns:
            New ModelRegistrationState with consul_confirmed=True.
        """
        new_status: EnumRegistrationStatus = (
            EnumRegistrationStatus.COMPLETE
            if self.postgres_confirmed
            else EnumRegistrationStatus.PARTIAL
        )
        return ModelRegistrationState(
            status=new_status,
            node_id=self.node_id,
            consul_confirmed=True,
            postgres_confirmed=self.postgres_confirmed,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def with_postgres_confirmed(self, event_id: UUID) -> ModelRegistrationState:
        """Transition state after PostgreSQL registration is confirmed.

        If Consul is already confirmed, status becomes "complete".
        Otherwise, status becomes "partial".

        Args:
            event_id: UUID of the event confirming PostgreSQL registration.

        Returns:
            New ModelRegistrationState with postgres_confirmed=True.
        """
        new_status: EnumRegistrationStatus = (
            EnumRegistrationStatus.COMPLETE
            if self.consul_confirmed
            else EnumRegistrationStatus.PARTIAL
        )
        return ModelRegistrationState(
            status=new_status,
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
            reason: The failure reason (validation_failed, consul_failed,
                postgres_failed, or both_failed).
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
            ...     status=EnumRegistrationStatus.FAILED, failure_reason="consul_failed"
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
