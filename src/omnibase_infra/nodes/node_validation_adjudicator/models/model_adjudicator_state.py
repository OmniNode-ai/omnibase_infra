# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Adjudicator State Model for Pure Reducer Pattern.

Immutable state model for the validation adjudicator FSM. Follows the
pure reducer pattern where state is passed in and returned from
reduce(), with no internal mutation.

FSM States:
    - collecting: Accumulating check results from executor
    - adjudicating: Applying scoring policy to collected results
    - verdict_emitted: Final verdict has been produced

Defensive Copy (Immutability) Pattern:
    All ``with_*`` transition methods return **new** frozen instances rather
    than mutating the current object.  This is a defensive-copy approach:
    callers always receive an independent snapshot, so no reference held
    elsewhere can observe a state change.  Because ``check_results`` is an
    immutable ``tuple``, a shallow copy (via Pydantic model construction)
    is sufficient -- there are no mutable containers to worry about.

Tracking:
    - OMN-2147: Validation Skeleton -- Orchestrator + Executor
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumAdjudicatorState, EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.models.validation.model_check_result import (
    ModelCheckResult,
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
        current_state: Value of the current FSM state (e.g. ``"collecting"``).
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


class ModelAdjudicatorState(BaseModel):
    """State model for the validation adjudicator reducer FSM.

    Immutable state passed to and returned from reduce().
    Follows the pure reducer pattern -- no internal state mutation.

    State transitions are performed via ``with_*`` methods that return
    new immutable instances.  Each ``with_*`` call produces a **defensive
    copy** (a brand-new frozen model) so that no caller holding a
    reference to a previous state can observe the mutation.  Because
    ``check_results`` is a tuple (immutable) and all other fields are
    primitives or ``None``, a shallow copy via normal Pydantic
    construction is sufficient.

    Attributes:
        status: Current FSM state.
        candidate_id: Candidate being validated (set when collecting starts).
        plan_id: Validation plan identifier (set when collecting starts).
        check_results: Accumulated check results from the executor.
        last_processed_event_id: Last processed event ID for idempotency.

    Raises:
        RuntimeHostError: On any illegal state transition, with a
            ``ModelInfraErrorContext`` attached for structured tracing
            and a sanitized error message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumAdjudicatorState = Field(
        default=EnumAdjudicatorState.COLLECTING,
        description="Current FSM state.",
    )
    candidate_id: UUID | None = Field(
        default=None,
        description="Candidate being validated.",
    )
    plan_id: UUID | None = Field(
        default=None,
        description="Validation plan identifier.",
    )
    check_results: tuple[ModelCheckResult, ...] = Field(
        default_factory=tuple,
        description="Accumulated check results from the executor.",
    )
    last_processed_event_id: UUID | None = Field(
        default=None,
        description="Last processed event ID for idempotency.",
    )

    # ------------------------------------------------------------------
    # State transition methods (pure -- return new instances)
    # ------------------------------------------------------------------

    def with_check_result(
        self, result: ModelCheckResult, event_id: UUID
    ) -> ModelAdjudicatorState:
        """Append a check result while staying in COLLECTING state.

        Args:
            result: Individual check result from the executor.
            event_id: UUID of the event triggering this update.

        Returns:
            New state with the check result appended.

        Raises:
            RuntimeHostError: If current state is not COLLECTING.
        """
        if self.status != EnumAdjudicatorState.COLLECTING:
            raise _make_transition_error(
                self.status.value, "collecting (append)", "COLLECTING"
            )
        return ModelAdjudicatorState(
            status=EnumAdjudicatorState.COLLECTING,
            candidate_id=self.candidate_id,
            plan_id=self.plan_id,
            check_results=(*self.check_results, result),
            last_processed_event_id=event_id,
        )

    def with_adjudication_started(self, event_id: UUID) -> ModelAdjudicatorState:
        """Transition: collecting -> adjudicating.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=ADJUDICATING.

        Raises:
            RuntimeHostError: If current state is not COLLECTING.
        """
        if not self.can_adjudicate():
            raise _make_transition_error(
                self.status.value, "adjudicating", "COLLECTING"
            )
        return ModelAdjudicatorState(
            status=EnumAdjudicatorState.ADJUDICATING,
            candidate_id=self.candidate_id,
            plan_id=self.plan_id,
            check_results=self.check_results,
            last_processed_event_id=event_id,
        )

    def with_verdict_emitted(self, event_id: UUID) -> ModelAdjudicatorState:
        """Transition: adjudicating -> verdict_emitted.

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=VERDICT_EMITTED.

        Raises:
            RuntimeHostError: If current state is not ADJUDICATING.
        """
        if not self.can_emit_verdict():
            raise _make_transition_error(
                self.status.value, "verdict_emitted", "ADJUDICATING"
            )
        return ModelAdjudicatorState(
            status=EnumAdjudicatorState.VERDICT_EMITTED,
            candidate_id=self.candidate_id,
            plan_id=self.plan_id,
            check_results=self.check_results,
            last_processed_event_id=event_id,
        )

    def with_reset(self, event_id: UUID) -> ModelAdjudicatorState:
        """Transition: verdict_emitted -> collecting (reset for next run).

        Args:
            event_id: UUID of the event triggering this transition.

        Returns:
            New state with status=COLLECTING and all data cleared.

        Raises:
            RuntimeHostError: If current state is not VERDICT_EMITTED.
        """
        if not self.can_reset():
            raise _make_transition_error(
                self.status.value, "collecting", "VERDICT_EMITTED"
            )
        return ModelAdjudicatorState(
            status=EnumAdjudicatorState.COLLECTING,
            candidate_id=None,
            plan_id=None,
            check_results=(),
            last_processed_event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Guard methods
    # ------------------------------------------------------------------

    def can_adjudicate(self) -> bool:
        """Check if adjudication can begin from the current state.

        Returns:
            True if the current state is COLLECTING.
        """
        return self.status == EnumAdjudicatorState.COLLECTING

    def can_emit_verdict(self) -> bool:
        """Check if a verdict can be emitted from the current state.

        Returns:
            True if the current state is ADJUDICATING.
        """
        return self.status == EnumAdjudicatorState.ADJUDICATING

    def can_reset(self) -> bool:
        """Check if the current state allows reset to collecting.

        Returns:
            True if the current state is VERDICT_EMITTED.
        """
        return self.status == EnumAdjudicatorState.VERDICT_EMITTED

    def is_duplicate_event(self, event_id: UUID) -> bool:
        """Check if an event has already been processed.

        Args:
            event_id: UUID of the event to check.

        Returns:
            True if this event_id matches the last processed event.

        Note:
            Only detects immediate replays (consecutive duplicate event IDs).
            Out-of-order redelivery is NOT detected. For Kafka consumers,
            use consumer-side deduplication with a bounded ID window if
            stronger guarantees are needed.
        """
        return self.last_processed_event_id == event_id


__all__: list[str] = ["ModelAdjudicatorState"]
