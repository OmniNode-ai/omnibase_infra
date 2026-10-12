# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that turns the adjudicator's verdict into the published result.

Consumes ``onex.evt.platform.validation-adjudication-completed.v1`` and produces the
``ModelValidationResultPublished`` the orchestrator publishes on
``onex.evt.platform.validation-result-published.v1``: the terminal event of the
validation pipeline. Pure COMPUTE: no I/O.

Related Tickets:
    - OMN-2147: Validation Skeleton Orchestrator + Executor
    - OMN-21056: validation pipeline reaches a published result
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import (
    ModelValidationAdjudicationCompleted,
    ModelValidationResultPublished,
)


class HandlerPublishResult:
    """Publishes the final validation result for one candidate."""

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
        """Behavioral classification: pure data transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelValidationAdjudicationCompleted
    ) -> ModelValidationResultPublished:
        """Build the terminal result from the adjudicator's verdict."""
        return ModelValidationResultPublished(
            candidate_id=request.candidate_id,
            plan_id=request.plan_id,
            verdict=request.verdict,
            score=request.score,
            total_checks=request.total_checks,
            passed_checks=request.passed_checks,
            failed_checks=request.failed_checks,
            skipped_checks=request.skipped_checks,
            blocking_failures=request.blocking_failures,
            quarantine_reasons=request.quarantine_reasons,
            adjudicated_at=request.adjudicated_at,
        )


__all__: list[str] = ["HandlerPublishResult"]
