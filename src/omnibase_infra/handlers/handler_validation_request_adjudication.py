# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that requests adjudication of a completed set of checks.

Consumes ``validation-checks-completed`` and returns the request the
adjudicator consumes from ``validation-adjudication-requested``. Pure
transformation: the orchestrator emits, the adjudicator decides.

Ticket: OMN-21056
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import (
    ModelExecutorResult,
    ModelValidationAdjudicationRequest,
)

logger = logging.getLogger(__name__)


class HandlerValidationRequestAdjudication:
    """Turns an executor result into an adjudication request."""

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-validation-request-adjudication"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level orchestration handler."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure transformation, no I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelExecutorResult
    ) -> ModelValidationAdjudicationRequest:
        """Carry the check results and the plan's threshold to the adjudicator.

        Args:
            request: The aggregated result of every check in the plan.

        Returns:
            The adjudication request for those results.
        """
        logger.info(
            "Requesting adjudication of %d check results for plan %s",
            len(request.check_results),
            request.plan_id,
        )
        return ModelValidationAdjudicationRequest(
            plan_id=request.plan_id,
            candidate_id=request.candidate_id,
            check_results=request.check_results,
            score_threshold=request.score_threshold,
        )


__all__: list[str] = ["HandlerValidationRequestAdjudication"]
