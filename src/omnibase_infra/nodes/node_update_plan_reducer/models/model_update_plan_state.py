# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Update Plan Lifecycle State Model for Pure Reducer Pattern.

Immutable state model for the update plan FSM. Follows the pure reducer
pattern where state is passed in and returned from reduce(), with no
internal mutation.

FSM States:
    - idle: No active plan — ready for new trigger
    - created: Plan created from impact analysis result
    - comment_posted: PR comment with impact summary posted
    - yaml_emitted: YAML plan emitted as structured artifact
    - closed: Plan fully processed and closed
    - waived: Plan skipped via explicit waiver

Defensive Copy (Immutability) Pattern:
    All ``with_*`` transition methods return **new** frozen instances rather
    than mutating the current object. This ensures no reference held elsewhere
    can observe a state change. Because every field is either a primitive or
    ``None``, a shallow copy via Pydantic model construction is sufficient.

Tracking:
    - OMN-3943: Task 6 — Update Plan REDUCER Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType, EnumUpdatePlanState
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
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


class ModelUpdatePlanState(BaseModel):
    """State model for the update plan reducer FSM.

    Immutable state passed to and returned from reduce().
    Follows the pure reducer pattern -- no internal state mutation.

    State transitions are performed via ``with_*`` methods that return
    new immutable instances. Each ``with_*`` call produces a **defensive
    copy** (a brand-new frozen model) so that no caller holding a
    reference to a previous state can observe the mutation.

    FSM transitions:
        idle -> created (on create_plan)
        created -> comment_posted (on post_comment)
        comment_posted -> yaml_emitted (on emit_yaml)
        yaml_emitted -> closed (on close)
        created -> waived (on waive, when merge_policy == "none")
        comment_posted -> waived (on waive)

    Attributes:
        status: Current FSM state.
        plan_id: Active plan identifier (set when a plan is created).
        last_processed_event_id: Last processed event ID for idempotency.

    Raises:
        RuntimeHostError: On any illegal state transition, with a
            ``ModelInfraErrorContext`` attached for structured tracing
            and a sanitized error message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumUpdatePlanState = Field(
        default=EnumUpdatePlanState.IDLE,
        description="Current FSM state.",
    )
    plan_id: UUID | None = Field(
        default=None,
        description="Active plan identifier.",
    )
    last_processed_event_id: UUID | None = Field(
        default=None,
        description="Last processed event ID for idempotency.",
    )

    # ------------------------------------------------------------------
    # State transition methods (pure — return new instances)
    # ------------------------------------------------------------------

    def with_plan_created(self, plan_id: UUID, event_id: UUID) -> ModelUpdatePlanState:
        """Transition: idle -> created.

        Args:
            plan_id: Unique identifier for the new plan.
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=CREATED.

        Raises:
            RuntimeHostError: If current state is not IDLE.
        """
        if not self.can_create_plan():
            raise _make_transition_error(self.status.value, "created", "IDLE")
        return ModelUpdatePlanState(
            status=EnumUpdatePlanState.CREATED,
            plan_id=plan_id,
            last_processed_event_id=event_id,
        )

    def with_comment_posted(self, event_id: UUID) -> ModelUpdatePlanState:
        """Transition: created -> comment_posted.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=COMMENT_POSTED.

        Raises:
            RuntimeHostError: If current state is not CREATED.
        """
        if self.status != EnumUpdatePlanState.CREATED:
            raise _make_transition_error(self.status.value, "comment_posted", "CREATED")
        return ModelUpdatePlanState(
            status=EnumUpdatePlanState.COMMENT_POSTED,
            plan_id=self.plan_id,
            last_processed_event_id=event_id,
        )

    def with_yaml_emitted(self, event_id: UUID) -> ModelUpdatePlanState:
        """Transition: comment_posted -> yaml_emitted.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=YAML_EMITTED.

        Raises:
            RuntimeHostError: If current state is not COMMENT_POSTED.
        """
        if self.status != EnumUpdatePlanState.COMMENT_POSTED:
            raise _make_transition_error(
                self.status.value, "yaml_emitted", "COMMENT_POSTED"
            )
        return ModelUpdatePlanState(
            status=EnumUpdatePlanState.YAML_EMITTED,
            plan_id=self.plan_id,
            last_processed_event_id=event_id,
        )

    def with_closed(self, event_id: UUID) -> ModelUpdatePlanState:
        """Transition: yaml_emitted -> closed.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=CLOSED.

        Raises:
            RuntimeHostError: If current state is not YAML_EMITTED.
        """
        if self.status != EnumUpdatePlanState.YAML_EMITTED:
            raise _make_transition_error(self.status.value, "closed", "YAML_EMITTED")
        return ModelUpdatePlanState(
            status=EnumUpdatePlanState.CLOSED,
            plan_id=self.plan_id,
            last_processed_event_id=event_id,
        )

    def with_waived(self, event_id: UUID) -> ModelUpdatePlanState:
        """Transition: created or comment_posted -> waived.

        Waiver is allowed from CREATED (e.g. none merge policy) or
        COMMENT_POSTED (e.g. operator explicitly waives after comment).

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=WAIVED.

        Raises:
            RuntimeHostError: If current state does not allow waiver.
        """
        if not self.can_waive():
            raise _make_transition_error(
                self.status.value, "waived", "CREATED or COMMENT_POSTED"
            )
        return ModelUpdatePlanState(
            status=EnumUpdatePlanState.WAIVED,
            plan_id=self.plan_id,
            last_processed_event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Guard predicates (public API — used externally)
    # ------------------------------------------------------------------

    def can_create_plan(self) -> bool:
        """Check if a new plan can be created from the current state.

        Returns:
            True if the current state is IDLE.
        """
        return self.status == EnumUpdatePlanState.IDLE

    def can_waive(self) -> bool:
        """Check if the plan can be waived from the current state.

        Waiver is allowed from CREATED (immediate waiver) or
        COMMENT_POSTED (waiver after comment).

        Returns:
            True if the current state is CREATED or COMMENT_POSTED.
        """
        return self.status in (
            EnumUpdatePlanState.CREATED,
            EnumUpdatePlanState.COMMENT_POSTED,
        )

    def is_terminal(self) -> bool:
        """Check if the current state is a terminal state.

        Returns:
            True if the current state is CLOSED or WAIVED.
        """
        return self.status in (
            EnumUpdatePlanState.CLOSED,
            EnumUpdatePlanState.WAIVED,
        )

    def is_duplicate_event(self, event_id: UUID) -> bool:
        """Check if an event has already been processed.

        Args:
            event_id: UUID of the event to check.

        Returns:
            True if this event_id matches the last processed event.

        Note:
            Only detects immediate replays (consecutive duplicate event IDs).
            Out-of-order redelivery is NOT detected.
        """
        return self.last_processed_event_id == event_id


__all__: list[str] = ["ModelUpdatePlanState"]
