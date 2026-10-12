# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that adjudicates a set of check results into a verdict.

Consumes ``validation-adjudication-requested`` and returns the
``ModelVerdict`` published on ``validation-adjudication-completed``. Pure fold
of the request into the adjudicator state, then the scoring policy of
``ModelVerdict.from_state``:

    1. Any REQUIRED check failure -> FAIL (hard block).
    2. Score < threshold but no hard blocks -> QUARANTINE (soft block).
    3. All REQUIRED pass and score >= threshold -> PASS.

Ticket: OMN-21056
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import ModelValidationAdjudicationRequest
from omnibase_infra.models.validation.model_adjudicator_state import (
    ModelAdjudicatorState,
)
from omnibase_infra.models.validation.model_verdict import (
    ModelVerdict,
)

logger = logging.getLogger(__name__)


class HandlerValidationAdjudicate:
    """Produces the PASS / FAIL / QUARANTINE verdict for a request."""

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-validation-adjudicate"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level handler."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure fold with no I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(self, request: ModelValidationAdjudicationRequest) -> ModelVerdict:
        """Score the request's check results under the scoring policy.

        Args:
            request: The check results to adjudicate and the PASS threshold.

        Returns:
            The verdict for the request's candidate and plan.
        """
        state = ModelAdjudicatorState(
            candidate_id=request.candidate_id,
            plan_id=request.plan_id,
            check_results=request.check_results,
        )
        verdict = ModelVerdict.from_state(
            state, score_threshold=request.score_threshold
        )
        logger.info(
            "Adjudicated candidate %s (plan %s): %s score=%.2f blocking=%s",
            verdict.candidate_id,
            verdict.plan_id,
            verdict.verdict.value,
            verdict.score,
            list(verdict.blocking_failures),
        )
        return verdict


__all__: list[str] = ["HandlerValidationAdjudicate"]
