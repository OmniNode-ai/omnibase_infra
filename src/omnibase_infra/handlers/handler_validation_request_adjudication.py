# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that turns executor results into an adjudication request.

Consumes ``onex.evt.platform.validation-checks-completed.v1`` (a
``ModelExecutorResult``) and produces the ``ModelValidationAdjudicationRequested``
that the orchestrator publishes on
``onex.evt.platform.validation-adjudication-requested.v1`` for the adjudicator.
Pure COMPUTE: no I/O.

Related Tickets:
    - OMN-2147: Validation Skeleton Orchestrator + Executor
    - OMN-21056: validation pipeline reaches a published result
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import (
    ModelExecutorResult,
    ModelValidationAdjudicationRequested,
)


class HandlerRequestAdjudication:
    """Wraps executor check results into an adjudication request."""

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
        """Behavioral classification: pure data transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelExecutorResult
    ) -> ModelValidationAdjudicationRequested:
        """Build the adjudication request for one executed plan."""
        return ModelValidationAdjudicationRequested(
            plan_id=request.plan_id,
            candidate_id=request.candidate_id,
            check_results=request.check_results,
        )


__all__: list[str] = ["HandlerRequestAdjudication"]
