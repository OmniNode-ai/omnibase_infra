# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Session Lifecycle State Model for Pure Reducer Pattern.

Immutable state model for the session lifecycle FSM. Follows the
pure reducer pattern where state is passed in and returned from
reduce(), with no internal mutation.

FSM States:
    - idle: No active run — waiting for pipeline start
    - run_created: Run document created, not yet active
    - run_active: Run is actively executing
    - run_ended: Run has completed or been terminated

Concurrency Model:
    - Each pipeline creates its own runs/{run_id}.json
    - session.json updates are append-only for recent_run_ids
    - active_run_id is advisory (for interactive sessions)
    - Multiple active runs allowed; destructive ops denied
      until /onex:set-active-run {run_id} disambiguates

Defensive Copy (Immutability) Pattern:
    All ``with_*`` transition methods return **new** frozen instances rather
    than mutating the current object.  This is a defensive-copy approach:
    callers always receive an independent snapshot, so no reference held
    elsewhere can observe a state change.  Because every field is either a
    primitive or ``None``, a shallow copy (via Pydantic model construction)
    is sufficient — there are no mutable containers to worry about.

Tracking:
    - OMN-2117: Canonical State Nodes
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType, EnumSessionLifecycleState
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.nodes.node_session_state_effect.models.model_run_context import (
    validate_run_id,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string


def _make_transition_error(
    current_state: str,
    target_state: str,
    required_state: str,
) -> RuntimeHostError:
    """Build a ``RuntimeHostError`` for an illegal FSM transition.

    The raw message is passed through ``sanitize_error_string`` before
    being stored on the error, and a ``ModelInfraErrorContext`` is
    attached for structured tracing.

    Args:
        current_state: Value of the current FSM state (e.g. ``"idle"``).
        target_state: The state the caller attempted to reach.
        required_state: The state that would have allowed the transition.

    Returns:
        A fully-constructed ``RuntimeHostError`` ready to raise.
    """
    raw_msg = (
        f"Cannot transition to {target_state} from state {current_state!r} "
        f"(requires {required_state})"
    )
    context = ModelInfraErrorContext.with_correlation(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="state_transition",
    )
    return RuntimeHostError(sanitize_error_string(raw_msg), context=context)


class ModelSessionLifecycleState(BaseModel):
    """State model for the session lifecycle reducer FSM.

    Immutable state passed to and returned from reduce().
    Follows the pure reducer pattern -- no internal state mutation.

    State transitions are performed via ``with_*`` methods that return
    new immutable instances.  Each ``with_*`` call produces a **defensive
    copy** (a brand-new frozen model) so that no caller holding a
    reference to a previous state can observe the mutation.  Because all
    fields are primitives or ``None``, a shallow copy via normal Pydantic
    construction is sufficient -- there are no mutable containers (such as
    ``dict`` or ``list``) that would require deep-copying.

    Attributes:
        status: Current FSM state.
        run_id: Active run identifier (set when a run is created).
        last_processed_event_id: Last processed event ID for idempotency.

    Raises:
        RuntimeHostError: On any illegal state transition, with a
            ``ModelInfraErrorContext`` attached for structured tracing
            and a sanitized error message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumSessionLifecycleState = Field(
        default=EnumSessionLifecycleState.IDLE,
        description="Current FSM state.",
    )
    run_id: str | None = Field(
        default=None,
        description="Active run identifier.",
    )
    last_processed_event_id: UUID | None = Field(
        default=None,
        description="Last processed event ID for idempotency.",
    )

    # ------------------------------------------------------------------
    # State transition methods (pure — return new instances)
    # ------------------------------------------------------------------

    def with_run_created(
        self, run_id: str, event_id: UUID
    ) -> ModelSessionLifecycleState:
        """Transition: idle -> run_created.

        Args:
            run_id: Unique identifier for the new run.
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=RUN_CREATED.

        Raises:
            RuntimeHostError: If current state is not IDLE.
            ValueError: If run_id is invalid (via ``validate_run_id``).
        """
        if not self.can_create_run():
            raise _make_transition_error(self.status.value, "run_created", "IDLE")
        validate_run_id(run_id)
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_CREATED,
            run_id=run_id,
            last_processed_event_id=event_id,
        )

    def with_run_activated(self, event_id: UUID) -> ModelSessionLifecycleState:
        """Transition: run_created -> run_active.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=RUN_ACTIVE.

        Raises:
            RuntimeHostError: If current state is not RUN_CREATED or
                ``run_id`` is missing.
        """
        # Check run_id first so the error message is specific.
        if self.status == EnumSessionLifecycleState.RUN_CREATED and not self.run_id:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="state_transition",
            )
            raise RuntimeHostError(
                sanitize_error_string("Cannot activate run: run_id is missing"),
                context=context,
            )
        if not self.can_activate_run():
            raise _make_transition_error(self.status.value, "run_active", "RUN_CREATED")
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ACTIVE,
            run_id=self.run_id,
            last_processed_event_id=event_id,
        )

    def with_run_ended(self, event_id: UUID) -> ModelSessionLifecycleState:
        """Transition: run_active -> run_ended.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=RUN_ENDED.

        Raises:
            RuntimeHostError: If current state is not RUN_ACTIVE or
                ``run_id`` is missing.
        """
        # Check run_id first so the error message is specific.
        if self.status == EnumSessionLifecycleState.RUN_ACTIVE and not self.run_id:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="state_transition",
            )
            raise RuntimeHostError(
                sanitize_error_string("Cannot end run: run_id is missing"),
                context=context,
            )
        if not self.can_end_run():
            raise _make_transition_error(self.status.value, "run_ended", "RUN_ACTIVE")
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ENDED,
            run_id=self.run_id,
            last_processed_event_id=event_id,
        )

    def with_reset(self, event_id: UUID) -> ModelSessionLifecycleState:
        """Transition: run_ended -> idle (reset for next run).

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=IDLE and run_id cleared.

        Raises:
            RuntimeHostError: If current state is not RUN_ENDED.
        """
        if not self.can_reset():
            raise _make_transition_error(self.status.value, "idle", "RUN_ENDED")
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.IDLE,
            run_id=None,
            last_processed_event_id=event_id,
        )

    def is_duplicate_event(self, event_id: UUID) -> bool:
        """Check if an event has already been processed.

        Args:
            event_id: UUID of the event to check.

        Returns:
            True if this event_id matches the last processed event.

        Note:
            Only detects immediate replays (consecutive duplicate event IDs).
            Out-of-order redelivery (e.g., event A, event B, event A replayed)
            is NOT detected. For Kafka consumers, use consumer-side deduplication
            with a bounded ID window if stronger guarantees are needed.
        """
        return self.last_processed_event_id == event_id

    def can_create_run(self) -> bool:
        """Check if a new run can be created from the current state.

        Returns:
            True if the current state is IDLE.
        """
        return self.status == EnumSessionLifecycleState.IDLE

    def can_activate_run(self) -> bool:
        """Check if the current run can be activated.

        Returns:
            True if the current state is RUN_CREATED and ``run_id`` is set.
            A state with status RUN_CREATED but no run_id is malformed and
            cannot be activated.
        """
        return (
            self.status == EnumSessionLifecycleState.RUN_CREATED
            and self.run_id is not None
        )

    def can_end_run(self) -> bool:
        """Check if the current run can be ended.

        Returns:
            True if the current state is RUN_ACTIVE and ``run_id`` is set.
            A state with status RUN_ACTIVE but no run_id is malformed and
            cannot transition to ended.
        """
        return (
            self.status == EnumSessionLifecycleState.RUN_ACTIVE
            and self.run_id is not None
        )

    def can_reset(self) -> bool:
        """Check if the current state allows reset to idle.

        Returns:
            True if the current state is RUN_ENDED.
        """
        return self.status == EnumSessionLifecycleState.RUN_ENDED


__all__: list[str] = ["ModelSessionLifecycleState"]
