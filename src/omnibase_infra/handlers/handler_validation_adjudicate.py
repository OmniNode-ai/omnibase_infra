# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that adjudicates check results into a PASS/FAIL/QUARANTINE verdict.

Consumes ``onex.evt.platform.validation-adjudication-requested.v1`` and produces the
``ModelVerdict`` published on ``onex.evt.platform.validation-adjudication-completed.v1``.
The scoring policy lives in ``ModelVerdict.from_state``; this handler only folds the
requested check results into an adjudicator state and applies it. Pure: no I/O.

Related Tickets:
    - OMN-2147: Validation Skeleton Orchestrator + Executor
    - OMN-21056: validation pipeline reaches a published result
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import ModelValidationAdjudicationRequested
from omnibase_infra.models.validation.model_adjudicator_state import (
    ModelAdjudicatorState,
)
from omnibase_infra.models.validation.model_verdict import ModelVerdict


class HandlerAdjudicate:
    """Produces the validation verdict for one candidate's check results."""

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-adjudicate"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level handler."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure data transformation."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelValidationAdjudicationRequested
    ) -> ModelVerdict:
        """Apply the scoring policy to the requested check results."""
        state = ModelAdjudicatorState(
            candidate_id=request.candidate_id,
            plan_id=request.plan_id,
            check_results=request.check_results,
        )
        return ModelVerdict.from_state(state, score_threshold=request.score_threshold)


__all__: list[str] = ["HandlerAdjudicate"]
