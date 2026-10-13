# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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

This module also hosts the orchestrator's three later workflow hops
(``HandlerRequestExecution``, ``HandlerRequestAdjudication``,
``HandlerPublishResult``). They live here because
``scripts/ci/infra-node-allowlist.txt`` is shrink-only: a new file under
``nodes/*/handlers/`` is refused, and this module is already allowlisted.

Related Tickets:
    - OMN-2147: Validation Skeleton Orchestrator + Executor
    - OMN-21056: Validation pipeline end to end on the lab
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
from omnibase_infra.models.validation.model_validation_adjudication_completed import (
    ModelValidationAdjudicationCompleted,
)
from omnibase_infra.models.validation.model_validation_adjudication_requested import (
    ModelValidationAdjudicationRequested,
)
from omnibase_infra.models.validation.model_validation_checks_completed import (
    ModelValidationChecksCompleted,
)
from omnibase_infra.models.validation.model_validation_execution_requested import (
    ModelValidationExecutionRequested,
)
from omnibase_infra.models.validation.model_validation_result_published import (
    ModelValidationResultPublished,
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
        >>> plan = await handler.handle(candidate)
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
        correlation_id: UUID | None = None,
    ) -> ModelValidationPlan:
        """Build a validation plan from a pattern candidate.

        Creates a ModelValidationPlan populated with the full MVP check
        catalog. All 12 checks are included; each is enabled by default.

        Args:
            candidate: The pattern candidate to build a plan for.
            correlation_id: Correlation ID for tracing. Auto-generated
                via ``uuid4()`` when ``None``.

        Returns:
            ModelValidationPlan with ordered checks from the MVP catalog.
        """
        correlation_id = correlation_id or uuid4()
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


class HandlerRequestExecution:
    """Turns a created validation plan into an execution request.

    Consumes the plan ``HandlerBuildPlan`` published (``validation-plan-created``)
    and returns the execution request the executor subscribes to. Pure compute:
    the returned event class selects the publish topic through the contract's
    ``published_events``.

    Related Tickets:
        - OMN-21056: Validation pipeline end to end on the lab
    """

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-request-execution"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level orchestration step."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelValidationPlan
    ) -> ModelValidationExecutionRequested:
        """Return the execution request carrying every field of ``request``."""
        return ModelValidationExecutionRequested.model_validate(request)


class HandlerRequestAdjudication:
    """Turns completed checks into an adjudication request.

    Consumes the executor's ``validation-checks-completed`` event and returns
    the adjudication request the adjudicator subscribes to. Pure compute.

    Related Tickets:
        - OMN-21056: Validation pipeline end to end on the lab
    """

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-request-adjudication"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level orchestration step."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelValidationChecksCompleted
    ) -> ModelValidationAdjudicationRequested:
        """Return the adjudication request for ``request``'s check results."""
        return ModelValidationAdjudicationRequested(
            plan_id=request.plan_id,
            candidate_id=request.candidate_id,
            score_threshold=request.score_threshold,
            check_results=request.check_results,
        )


class HandlerPublishResult:
    """Turns an adjudicated verdict into the published validation result.

    Consumes the adjudicator's ``validation-adjudication-completed`` verdict and
    returns the pipeline's terminal ``validation-result-published`` event. Pure
    compute.

    Related Tickets:
        - OMN-21056: Validation pipeline end to end on the lab
    """

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-publish-result"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level orchestration step."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelValidationAdjudicationCompleted
    ) -> ModelValidationResultPublished:
        """Return the published result carrying every field of the verdict."""
        return ModelValidationResultPublished.model_validate(request)


__all__: list[str] = [
    "HandlerBuildPlan",
    "HandlerPublishResult",
    "HandlerRequestAdjudication",
    "HandlerRequestExecution",
]
