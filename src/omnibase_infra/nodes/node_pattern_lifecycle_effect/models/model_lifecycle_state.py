# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Lifecycle state model for pattern tier tracking.

Tracks the current promotion tier, consecutive verdict counts, and
total validations for a single pattern.  Provides a pure ``with_verdict``
method for computing the next state given a validation verdict.

Promotion ladder:
    OBSERVED -> SUGGESTED -> SHADOW_APPLY -> PROMOTED -> DEFAULT

Auto-rollback rules:
    - 2 consecutive FAIL verdicts: demote one tier
    - 3 consecutive FAIL verdicts: suppress the pattern
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumLifecycleTier, EnumValidationVerdict


class ModelLifecycleState(BaseModel):
    """Lifecycle state for a single pattern.

    This model is frozen (immutable).  Use :meth:`with_verdict` to produce
    a new state reflecting the outcome of a validation verdict.

    Attributes:
        pattern_id: Identifier of the pattern being tracked.
        current_tier: Current promotion tier.
        consecutive_pass_count: Consecutive PASS verdicts (reset on FAIL).
        consecutive_fail_count: Consecutive FAIL verdicts (reset on PASS).
        total_validations: Total number of verdicts applied.
        last_verdict: Most recent verdict applied, or None if never evaluated.
        last_updated: Timestamp of the most recent verdict application.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pattern_id: UUID = Field(
        ...,
        description="Identifier of the pattern being tracked.",
    )
    current_tier: EnumLifecycleTier = Field(
        default=EnumLifecycleTier.OBSERVED,
        description="Current promotion tier.",
    )
    consecutive_pass_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive PASS verdicts (reset on FAIL).",
    )
    consecutive_fail_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive FAIL verdicts (reset on PASS).",
    )
    total_validations: int = Field(
        default=0,
        ge=0,
        description="Total number of verdicts applied.",
    )
    last_verdict: EnumValidationVerdict | None = Field(
        default=None,
        description="Most recent verdict applied, or None if never evaluated.",
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Timestamp of the most recent verdict application.",
    )

    # ------------------------------------------------------------------
    # Pure state transitions
    # ------------------------------------------------------------------

    def with_verdict(self, verdict: EnumValidationVerdict) -> ModelLifecycleState:
        """Apply a verdict and return the next lifecycle state.

        Rules:
            - **PASS**: Reset fail counter, increment pass counter.
              Promote if ``consecutive_pass_count >= 2`` and tier allows it.
            - **FAIL**: Reset pass counter, increment fail counter.
              Demote if ``consecutive_fail_count >= 2``.
              Suppress if ``consecutive_fail_count >= 3``.
            - **QUARANTINE**: Record the verdict without changing the tier.

        Args:
            verdict: The validation verdict to apply.

        Returns:
            A new ``ModelLifecycleState`` reflecting the transition.
        """
        now = datetime.now(tz=UTC)
        new_total = self.total_validations + 1

        if verdict == EnumValidationVerdict.PASS:
            return self._apply_pass(now, new_total)
        if verdict == EnumValidationVerdict.FAIL:
            return self._apply_fail(now, new_total)
        # QUARANTINE -- record but do not change tier
        return self.model_copy(
            update={
                "last_verdict": verdict,
                "last_updated": now,
                "total_validations": new_total,
            },
        )

    def can_promote(self) -> bool:
        """Return True if the current tier supports promotion.

        Delegates to :meth:`EnumLifecycleTier.can_promote`.
        """
        return self.current_tier.can_promote()

    def is_suppressed(self) -> bool:
        """Return True if the pattern has been suppressed."""
        return self.current_tier == EnumLifecycleTier.SUPPRESSED

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_pass(self, now: datetime, new_total: int) -> ModelLifecycleState:
        """Apply a PASS verdict.

        Increments the pass counter, resets the fail counter, and promotes
        the tier when ``consecutive_pass_count`` reaches 2 (the count
        *after* incrementing).
        """
        new_pass_count = self.consecutive_pass_count + 1
        new_tier = self.current_tier

        # Promote after 2 consecutive passes (if tier allows promotion)
        if new_pass_count >= 2 and self.current_tier.can_promote():
            new_tier = self.current_tier.promoted()
            # Reset pass counter after promotion so the next tier
            # requires its own 2-pass streak.
            new_pass_count = 0

        return self.model_copy(
            update={
                "current_tier": new_tier,
                "consecutive_pass_count": new_pass_count,
                "consecutive_fail_count": 0,
                "total_validations": new_total,
                "last_verdict": EnumValidationVerdict.PASS,
                "last_updated": now,
            },
        )

    def _apply_fail(self, now: datetime, new_total: int) -> ModelLifecycleState:
        """Apply a FAIL verdict.

        Increments the fail counter, resets the pass counter.
        - At 2 consecutive fails: demote one tier.
        - At 3 consecutive fails: suppress the pattern.
        """
        new_fail_count = self.consecutive_fail_count + 1
        new_tier = self.current_tier

        if new_fail_count >= 3:
            new_tier = EnumLifecycleTier.SUPPRESSED
        elif new_fail_count >= 2:
            new_tier = self.current_tier.demoted()

        return self.model_copy(
            update={
                "current_tier": new_tier,
                "consecutive_pass_count": 0,
                "consecutive_fail_count": new_fail_count,
                "total_validations": new_total,
                "last_verdict": EnumValidationVerdict.FAIL,
                "last_updated": now,
            },
        )


__all__: list[str] = ["ModelLifecycleState"]
