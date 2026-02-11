# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation verdict model emitted by the adjudicator reducer.

The verdict is the final output of the validation pipeline for a given
candidate.  It summarises all check results into a single
PASS / FAIL / QUARANTINE decision together with scoring metadata.

Scoring Policy:
    1. Any REQUIRED check failure -> FAIL (hard block).
    2. Score < threshold but no hard blocks -> QUARANTINE (soft block).
    3. All REQUIRED pass and score >= threshold -> PASS.

The ``from_state`` classmethod encapsulates this scoring logic so that
callers only need to supply the adjudicator state and a threshold.

Tracking:
    - OMN-2147: Validation Skeleton -- Orchestrator + Executor
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import (
    EnumCheckSeverity,
    EnumInfraTransportType,
    EnumValidationVerdict,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_validation_adjudicator.models.model_adjudicator_state import (
        ModelAdjudicatorState,
    )


class ModelVerdict(BaseModel):
    """Validation verdict produced by the adjudicator reducer.

    Summarises the collected check results into a single actionable
    decision with scoring metadata.

    Attributes:
        candidate_id: Candidate that was validated.
        plan_id: Validation plan used.
        verdict: Final decision (PASS / FAIL / QUARANTINE).
        score: Weighted pass ratio in [0.0, 1.0].
        total_checks: Total number of checks executed.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        skipped_checks: Number of checks that were skipped.
        blocking_failures: Check codes that caused a FAIL verdict.
        quarantine_reasons: Human-readable reasons for QUARANTINE.
        adjudicated_at: Timestamp when the verdict was produced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    candidate_id: UUID = Field(..., description="Candidate that was validated.")
    plan_id: UUID = Field(..., description="Validation plan used.")
    verdict: EnumValidationVerdict = Field(
        ..., description="Final decision (PASS / FAIL / QUARANTINE)."
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted pass ratio in [0.0, 1.0]."
    )
    total_checks: int = Field(..., ge=0, description="Total number of checks executed.")
    passed_checks: int = Field(..., ge=0, description="Number of checks that passed.")
    failed_checks: int = Field(..., ge=0, description="Number of checks that failed.")
    skipped_checks: int = Field(
        ..., ge=0, description="Number of checks that were skipped."
    )
    blocking_failures: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Check codes that caused a FAIL verdict.",
    )
    quarantine_reasons: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Human-readable reasons for QUARANTINE.",
    )
    adjudicated_at: datetime = Field(
        ..., description="Timestamp when the verdict was produced."
    )

    @classmethod
    def from_state(
        cls,
        state: ModelAdjudicatorState,
        score_threshold: float = 0.8,
    ) -> ModelVerdict:
        """Compute a verdict from the adjudicator's accumulated state.

        Scoring policy:
            1. Any REQUIRED check that failed -> FAIL with blocking_failures.
            2. Score < ``score_threshold`` but no hard blocks -> QUARANTINE.
            3. All REQUIRED pass and score >= threshold -> PASS.

        The score is computed as the ratio of non-skipped checks that
        passed to total non-skipped checks.  Skipped checks do not
        contribute to the score.

        Args:
            state: Current adjudicator state with accumulated check results.
            score_threshold: Minimum score for a PASS verdict (default 0.8).

        Returns:
            A fully-populated ``ModelVerdict`` instance.

        Raises:
            RuntimeHostError: If ``state.candidate_id`` or ``state.plan_id`` is None.
        """
        if state.candidate_id is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="adjudicate_verdict",
            )
            raise RuntimeHostError(
                "Cannot produce verdict: candidate_id is None", context=context
            )
        if state.plan_id is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="adjudicate_verdict",
            )
            raise RuntimeHostError(
                "Cannot produce verdict: plan_id is None", context=context
            )

        results = state.check_results

        passed = sum(1 for r in results if r.passed and not r.skipped)
        failed = sum(1 for r in results if not r.passed and not r.skipped)
        skipped = sum(1 for r in results if r.skipped)
        total = len(results)

        # Score: ratio of passed to non-skipped checks (0.0 if all skipped)
        non_skipped = passed + failed
        score = passed / non_skipped if non_skipped > 0 else 1.0

        # Identify blocking failures (REQUIRED severity that failed)
        blocking = tuple(r.check_code for r in results if r.is_blocking_failure())

        # Determine verdict
        if blocking:
            verdict = EnumValidationVerdict.FAIL
            quarantine_reasons: tuple[str, ...] = ()
        elif score < score_threshold:
            verdict = EnumValidationVerdict.QUARANTINE
            # Build quarantine reasons from RECOMMENDED failures
            quarantine_reasons = tuple(
                f"{r.check_code}: {r.message}" if r.message else r.check_code
                for r in results
                if not r.passed
                and not r.skipped
                and r.severity == EnumCheckSeverity.RECOMMENDED
            )
            if not quarantine_reasons:
                quarantine_reasons = (
                    f"Score {score:.2f} below threshold {score_threshold:.2f}",
                )
        else:
            verdict = EnumValidationVerdict.PASS
            quarantine_reasons = ()

        return cls(
            candidate_id=state.candidate_id,
            plan_id=state.plan_id,
            verdict=verdict,
            score=score,
            total_checks=total,
            passed_checks=passed,
            failed_checks=failed,
            skipped_checks=skipped,
            blocking_failures=blocking,
            quarantine_reasons=quarantine_reasons,
            adjudicated_at=datetime.now(UTC),
        )


__all__: list[str] = ["ModelVerdict"]
