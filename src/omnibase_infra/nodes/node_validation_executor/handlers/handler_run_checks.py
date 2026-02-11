# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for executing validation checks from a plan.

MVP skeleton: iterates through planned checks and produces
``ModelCheckResult`` entries. Actual subprocess execution (mypy, ruff,
pytest) will be wired in a follow-up ticket.

Ticket: OMN-2147
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import (
    ModelCheckResult,
    ModelExecutorResult,
    ModelValidationPlan,
)

logger = logging.getLogger(__name__)


class HandlerRunChecks:
    """Execute all checks defined in a validation plan.

    MVP skeleton implementation that records check results without actually
    running subprocess commands. Each enabled check in the plan produces a
    ``ModelCheckResult`` with ``passed=True``. Disabled checks produce a
    result with ``skipped=True``.

    Actual subprocess execution (``mypy``, ``ruff``, ``pytest``) will be
    wired in a follow-up ticket.
    """

    def __init__(self) -> None:
        """Initialize the handler (stateless)."""

    @property
    def handler_id(self) -> str:
        """Return the unique handler identifier.

        Returns:
            Handler identifier string.
        """
        return "handler-run-checks"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for subprocess I/O.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler performs
            infrastructure-level I/O (subprocess execution).
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting subprocess execution.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs
            side-effecting operations (runs external processes).
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        plan: ModelValidationPlan,
        correlation_id: UUID,
    ) -> ModelExecutorResult:
        """Execute all checks in the validation plan.

        MVP: Skeleton implementation that records check results without
        running actual subprocesses. Enabled checks are recorded as
        passed; disabled checks are recorded as skipped.

        Actual subprocess execution (mypy, ruff, pytest) will be wired
        in a follow-up ticket.

        Args:
            plan: Validation plan containing the checks to execute.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Aggregated executor result with all check outcomes.
        """
        logger.info(
            "Executing validation plan %s with %d checks (correlation_id=%s)",
            plan.plan_id,
            len(plan.checks),
            correlation_id,
        )

        start_time = time.monotonic()
        results: list[ModelCheckResult] = []

        for planned_check in plan.checks:
            check_start = time.monotonic()

            if not planned_check.enabled:
                result = ModelCheckResult(
                    check_code=planned_check.check_code,
                    label=planned_check.label,
                    severity=planned_check.severity,
                    passed=False,
                    skipped=True,
                    message="Check disabled in plan.",
                    duration_ms=0.0,
                    executed_at=datetime.now(tz=UTC),
                )
            else:
                # MVP skeleton: record the check as passed.
                # In the follow-up, this will run the actual command
                # via asyncio.create_subprocess_exec and capture output.
                check_duration_ms = (time.monotonic() - check_start) * 1000.0
                result = ModelCheckResult(
                    check_code=planned_check.check_code,
                    label=planned_check.label,
                    severity=planned_check.severity,
                    passed=True,
                    skipped=False,
                    message=f"Skeleton pass for '{planned_check.command}'.",
                    duration_ms=check_duration_ms,
                    executed_at=datetime.now(tz=UTC),
                )

            results.append(result)
            logger.debug(
                "Check %s (%s): passed=%s skipped=%s",
                planned_check.check_code,
                planned_check.label,
                result.passed,
                result.skipped,
            )

        total_duration_ms = (time.monotonic() - start_time) * 1000.0

        executor_result = ModelExecutorResult(
            plan_id=plan.plan_id,
            candidate_id=plan.candidate_id,
            check_results=tuple(results),
            total_duration_ms=total_duration_ms,
        )

        logger.info(
            "Validation plan %s complete: %d passed, %d failed, %.1f ms total",
            plan.plan_id,
            executor_result.pass_count,
            executor_result.fail_count,
            total_duration_ms,
        )

        return executor_result


__all__: list[str] = ["HandlerRunChecks"]
