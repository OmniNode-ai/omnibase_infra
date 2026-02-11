# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that builds a validation plan from a pattern candidate.

Receives a ModelPatternCandidate and produces a ModelValidationPlan containing
the ordered set of checks from the MVP check catalog. This is a pure COMPUTE
handler -- no I/O, no side effects.

Check Catalog (MVP):
    CHECK-PY-001    Typecheck (mypy)                     required
    CHECK-PY-002    Lint/format (ruff)                   required
    CHECK-TEST-001  Unit tests (fast)                    required
    CHECK-TEST-002  Targeted integration tests           recommended
    CHECK-VAL-001   Deterministic replay sanity          recommended
    CHECK-VAL-002   Artifact completeness                required
    CHECK-RISK-001  Sensitive paths -> stricter bar       required
    CHECK-RISK-002  Diff size threshold                  recommended
    CHECK-RISK-003  Unsafe operations detector           required
    CHECK-OUT-001   CI equivalent pass rate              required
    CHECK-COST-001  Token delta vs baseline              informational
    CHECK-TIME-001  Wall-clock delta vs baseline         informational

Coroutine Safety:
    This handler is stateless and coroutine-safe for concurrent calls
    with different candidate instances.

Related Tickets:
    - OMN-2147: Validation Skeleton Orchestrator + Executor
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_infra.enums import (
    EnumCheckSeverity,
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_validation_orchestrator.models import (
    ModelPatternCandidate,
    ModelPlannedCheck,
    ModelValidationPlan,
)

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MVP check catalog -- static definition of all planned checks.
# Each tuple is (check_code, label, severity).
# ---------------------------------------------------------------------------
_CHECK_CATALOG: tuple[tuple[str, str, EnumCheckSeverity], ...] = (
    ("CHECK-PY-001", "Typecheck (mypy)", EnumCheckSeverity.REQUIRED),
    ("CHECK-PY-002", "Lint/format (ruff)", EnumCheckSeverity.REQUIRED),
    ("CHECK-TEST-001", "Unit tests (fast)", EnumCheckSeverity.REQUIRED),
    ("CHECK-TEST-002", "Targeted integration tests", EnumCheckSeverity.RECOMMENDED),
    ("CHECK-VAL-001", "Deterministic replay sanity", EnumCheckSeverity.RECOMMENDED),
    ("CHECK-VAL-002", "Artifact completeness", EnumCheckSeverity.REQUIRED),
    ("CHECK-RISK-001", "Sensitive paths -> stricter bar", EnumCheckSeverity.REQUIRED),
    ("CHECK-RISK-002", "Diff size threshold", EnumCheckSeverity.RECOMMENDED),
    ("CHECK-RISK-003", "Unsafe operations detector", EnumCheckSeverity.REQUIRED),
    ("CHECK-OUT-001", "CI equivalent pass rate", EnumCheckSeverity.REQUIRED),
    ("CHECK-COST-001", "Token delta vs baseline", EnumCheckSeverity.INFORMATIONAL),
    ("CHECK-TIME-001", "Wall-clock delta vs baseline", EnumCheckSeverity.INFORMATIONAL),
)


class HandlerBuildPlan:
    """Builds a validation plan from a pattern candidate.

    Pure COMPUTE handler -- receives a ModelPatternCandidate and returns a
    ModelValidationPlan populated with the full MVP check catalog. No I/O
    is performed; all checks are statically defined.

    Attributes:
        handler_id: Unique handler identifier.
        handler_type: Architectural role (NODE_HANDLER).
        handler_category: Behavioral classification (COMPUTE).

    Example:
        >>> from uuid import uuid4
        >>> handler = HandlerBuildPlan()
        >>> candidate = ModelPatternCandidate(
        ...     candidate_id=uuid4(),
        ...     pattern_id="pattern-001",
        ...     source_path="/src/my_module",
        ... )
        >>> plan = await handler.handle(candidate, correlation_id=uuid4())
        >>> len(plan.checks)
        12
    """

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-build-plan"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role classification for this handler.

        Returns NODE_HANDLER because this handler processes node-level
        orchestration events (building validation plans).
        """
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification for this handler.

        Returns COMPUTE because this handler performs pure data
        transformation with no I/O or side effects.
        """
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        candidate: ModelPatternCandidate,
        correlation_id: UUID,
    ) -> ModelValidationPlan:
        """Build a validation plan from a pattern candidate.

        Creates a ModelValidationPlan populated with the full MVP check
        catalog. All 12 checks are included; each is enabled by default.

        Args:
            candidate: The pattern candidate to build a plan for.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelValidationPlan with ordered checks from the MVP catalog.
        """
        plan_id = uuid4()

        checks = tuple(
            ModelPlannedCheck(
                check_code=check_code,
                label=label,
                severity=severity,
                enabled=True,
            )
            for check_code, label, severity in _CHECK_CATALOG
        )

        plan = ModelValidationPlan(
            plan_id=plan_id,
            candidate_id=candidate.candidate_id,
            checks=checks,
            score_threshold=0.8,
        )

        logger.info(
            "Built validation plan with %d checks for candidate %s",
            len(checks),
            candidate.candidate_id,
            extra={
                "plan_id": str(plan_id),
                "candidate_id": str(candidate.candidate_id),
                "correlation_id": str(correlation_id),
                "check_count": len(checks),
                "required_count": sum(
                    1 for c in checks if c.severity == EnumCheckSeverity.REQUIRED
                ),
            },
        )

        return plan


__all__: list[str] = ["HandlerBuildPlan"]
