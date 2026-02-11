# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Shared validation pipeline model — aggregated executor result."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_check_result import (
    ModelCheckResult,
)


class ModelExecutorResult(BaseModel):
    """Aggregated result of all check executions.

    Attributes:
        plan_id: Reference to the validation plan that was executed.
        candidate_id: Reference to the pattern candidate.
        check_results: Tuple of individual check results.
        total_duration_ms: Total execution duration in milliseconds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    plan_id: UUID = Field(..., description="Reference to the validation plan.")
    candidate_id: UUID = Field(..., description="Reference to the pattern candidate.")
    check_results: tuple[ModelCheckResult, ...] = Field(
        default_factory=tuple, description="Individual check results."
    )
    total_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total execution duration in ms."
    )

    @property
    def all_required_passed(self) -> bool:
        """Return True if all required checks passed."""
        from omnibase_infra.enums import EnumCheckSeverity

        return all(
            r.passed or r.skipped
            for r in self.check_results
            if r.severity == EnumCheckSeverity.REQUIRED
        )

    @property
    def has_blocking_failures(self) -> bool:
        """Return True if any required checks failed."""
        return any(r.is_blocking_failure() for r in self.check_results)

    @property
    def pass_count(self) -> int:
        """Number of checks that passed."""
        return sum(1 for r in self.check_results if r.passed)

    @property
    def fail_count(self) -> int:
        """Number of checks that failed (not skipped)."""
        return sum(1 for r in self.check_results if not r.passed and not r.skipped)

    def __bool__(self) -> bool:
        """Allow boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns True only when
            all required checks passed.
        """
        return self.all_required_passed


__all__: list[str] = ["ModelExecutorResult"]
