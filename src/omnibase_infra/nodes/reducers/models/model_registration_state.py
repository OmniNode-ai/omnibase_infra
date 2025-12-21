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

Related:
    - RegistrationReducer: Pure reducer that uses this state model
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md: Architecture design
    - OMN-889: Infrastructure MVP
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# Type alias for status literals
RegistrationStatus = Literal["idle", "pending", "partial", "complete", "failed"]

# Type alias for failure reason literals
FailureReason = Literal[
    "validation_failed", "consul_failed", "postgres_failed", "both_failed"
]


class ModelRegistrationState(BaseModel):
    """State model for the dual registration reducer workflow.

    Immutable state passed to and returned from reduce().
    Follows pure reducer pattern - no internal state mutation.

    The state tracks the current workflow status and confirmation state
    for both Consul and PostgreSQL backends. State transitions are
    performed via `with_*` methods that return new immutable instances.

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

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RegistrationStatus = Field(
        default="idle",
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
            status="pending",
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
        new_status: RegistrationStatus = (
            "complete" if self.postgres_confirmed else "partial"
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
        new_status: RegistrationStatus = (
            "complete" if self.consul_confirmed else "partial"
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
            status="failed",
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


__all__ = ["ModelRegistrationState"]
