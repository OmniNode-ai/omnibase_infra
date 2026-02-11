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

Tracking:
    - OMN-2117: Canonical State Nodes
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumSessionLifecycleState


class ModelSessionLifecycleState(BaseModel):
    """State model for the session lifecycle reducer FSM.

    Immutable state passed to and returned from reduce().
    Follows the pure reducer pattern — no internal state mutation.

    State transitions are performed via ``with_*`` methods that return
    new immutable instances.

    Attributes:
        status: Current FSM state.
        run_id: Active run identifier (set when a run is created).
        last_processed_event_id: Last processed event ID for idempotency.
        failure_reason: Reason for unexpected state (informational).
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
    failure_reason: str | None = Field(
        default=None,
        description="Reason for unexpected state (informational).",
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
            ValueError: If current state is not IDLE.
        """
        if not self.can_create_run():
            msg = f"Cannot create run from state {self.status.value!r} (requires IDLE)"
            raise ValueError(msg)
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_CREATED,
            run_id=run_id,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def with_run_activated(self, event_id: UUID) -> ModelSessionLifecycleState:
        """Transition: run_created -> run_active.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=RUN_ACTIVE.

        Raises:
            ValueError: If current state is not RUN_CREATED.
        """
        if not self.can_activate_run():
            msg = f"Cannot activate run from state {self.status.value!r} (requires RUN_CREATED)"
            raise ValueError(msg)
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ACTIVE,
            run_id=self.run_id,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def with_run_ended(self, event_id: UUID) -> ModelSessionLifecycleState:
        """Transition: run_active -> run_ended.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=RUN_ENDED.

        Raises:
            ValueError: If current state is not RUN_ACTIVE.
        """
        if not self.can_end_run():
            msg = (
                f"Cannot end run from state {self.status.value!r} (requires RUN_ACTIVE)"
            )
            raise ValueError(msg)
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.RUN_ENDED,
            run_id=self.run_id,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def with_reset(self, event_id: UUID) -> ModelSessionLifecycleState:
        """Transition: run_ended -> idle (reset for next run).

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=IDLE and run_id cleared.

        Raises:
            ValueError: If current state is not RUN_ENDED.
        """
        if not self.can_reset():
            msg = f"Cannot reset from state {self.status.value!r} (requires RUN_ENDED)"
            raise ValueError(msg)
        return ModelSessionLifecycleState(
            status=EnumSessionLifecycleState.IDLE,
            run_id=None,
            last_processed_event_id=event_id,
            failure_reason=None,
        )

    def is_duplicate_event(self, event_id: UUID) -> bool:
        """Check if an event has already been processed.

        Args:
            event_id: UUID of the event to check.

        Returns:
            True if this event_id matches the last processed event.
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
            True if the current state is RUN_CREATED.
        """
        return self.status == EnumSessionLifecycleState.RUN_CREATED

    def can_end_run(self) -> bool:
        """Check if the current run can be ended.

        Returns:
            True if the current state is RUN_ACTIVE.
        """
        return self.status == EnumSessionLifecycleState.RUN_ACTIVE

    def can_reset(self) -> bool:
        """Check if the current state allows reset to idle.

        Returns:
            True if the current state is RUN_ENDED.
        """
        return self.status == EnumSessionLifecycleState.RUN_ENDED


__all__: list[str] = ["ModelSessionLifecycleState"]
