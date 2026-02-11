# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""RRH result model composing shared primitives.

``ModelRRHResult`` composes ``ModelRuleCheckResult`` (per-rule outcomes)
and ``EnumVerdict`` (aggregate verdict) so that dashboards can consume
both architecture-validation and RRH results through the same types.

Verdict Semantics:
    PASS  — all applicable checks passed
    WARN  — non-critical issues (e.g. missing optional toolchain)
    FAIL  — critical violation blocking release
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from omnibase_infra.diagnostics.enum_verdict import EnumVerdict
from omnibase_infra.nodes.architecture_validator.models.model_rule_check_result import (
    ModelRuleCheckResult,
)


class ModelRRHResult(BaseModel):
    """Aggregate RRH validation result.

    Composes the same ``ModelRuleCheckResult`` and ``EnumVerdict`` types
    used by the architecture validation pipeline so that dashboards and
    downstream consumers can process both uniformly.

    Attributes:
        checks: Individual rule check results (RRH-1001 through RRH-1701).
        verdict: Aggregate verdict derived from worst check outcome.
        profile_name: Name of the profile used for this validation.
        ticket_id: Ticket identifier this RRH was run for (empty if N/A).
        repo_name: Repository name this RRH was run against.
        correlation_id: Distributed tracing correlation ID.
        evaluated_at: Timezone-aware timestamp when validation completed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    checks: tuple[ModelRuleCheckResult, ...] = Field(
        ..., description="Per-rule check results."
    )
    verdict: EnumVerdict = Field(..., description="Aggregate verdict.")
    profile_name: str = Field(..., description="Profile used for validation.")
    ticket_id: str = Field(default="", description="Ticket identifier.")
    repo_name: str = Field(default="", description="Repository name.")
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing."
    )
    evaluated_at: AwareDatetime = Field(
        ..., description="Timezone-aware evaluation timestamp."
    )

    def __bool__(self) -> bool:
        """Allow boolean context: True when verdict is PASS.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``verdict`` is ``EnumVerdict.PASS``. Differs from typical
            Pydantic behavior where any populated model is truthy.
        """
        return self.verdict == EnumVerdict.PASS

    @property
    def failed_checks(self) -> tuple[ModelRuleCheckResult, ...]:
        """Return only checks that represent violations."""
        return tuple(c for c in self.checks if c.is_violation())

    @property
    def applicable_checks(self) -> tuple[ModelRuleCheckResult, ...]:
        """Return only checks that were actually evaluated (not skipped)."""
        return tuple(c for c in self.checks if c.is_applicable())


__all__: list[str] = ["ModelRRHResult"]
