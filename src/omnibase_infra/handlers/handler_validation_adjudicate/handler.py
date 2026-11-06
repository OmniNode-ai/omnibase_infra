# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that adjudicates a plan's check results into a verdict.

Consumes the orchestrator's ``validation-adjudication-requested`` event, runs
the adjudicator FSM from ``collecting`` to ``adjudicating`` over the request's
check results, applies the scoring policy in ``ModelVerdict.from_state``, and
returns the verdict. The reducer's contract declares the returned event class
under ``published_events``, so the runtime publishes it as an event rather than
treating it as a projection.

The node had a state model and a scoring policy but no handler: no subscription
could ever have reached them. This module lives in the shared infra handlers
package, like the CI live-contact handler, because
``scripts/ci/infra-node-allowlist.txt`` is shrink-only and refuses a new file
under ``nodes/*/handlers/``.

Related Tickets:
    - OMN-2147: Validation Skeleton -- Orchestrator + Executor
    - OMN-21056: Validation pipeline end to end on the lab
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation.model_adjudicator_state import (
    ModelAdjudicatorState,
)
from omnibase_infra.models.validation.model_validation_adjudication_completed import (
    ModelValidationAdjudicationCompleted,
)
from omnibase_infra.models.validation.model_validation_adjudication_requested import (
    ModelValidationAdjudicationRequested,
)
from omnibase_infra.models.validation.model_verdict import (
    ModelVerdict,
)

logger = logging.getLogger(__name__)


class HandlerAdjudicate:
    """Adjudicates the check results of one validation plan."""

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-adjudicate"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level reducer step."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: the verdict timestamp reads the clock."""
        return EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE

    async def handle(
        self,
        request: ModelValidationAdjudicationRequested,
        correlation_id: UUID | None = None,
    ) -> ModelValidationAdjudicationCompleted:
        """Return the verdict for ``request``'s check results.

        The runtime hands the handler the typed request alone, so
        ``correlation_id`` is optional; it is only used to tag error context.
        The request's plan id is the FSM event id: a redelivered request is
        the same event.

        Args:
            request: Check results of one executed validation plan.
            correlation_id: Correlation ID for error context, when the caller
                has one.

        Returns:
            The verdict, published as ``validation-adjudication-completed``.

        Raises:
            RuntimeHostError: If the FSM refuses the transition.
        """
        collecting = ModelAdjudicatorState(
            candidate_id=request.candidate_id,
            plan_id=request.plan_id,
            check_results=request.check_results,
        )
        adjudicating = collecting.with_adjudication_started(
            request.plan_id, correlation_id
        )
        verdict = ModelVerdict.from_state(
            adjudicating,
            score_threshold=request.score_threshold,
            correlation_id=correlation_id,
        )
        logger.info(
            "Adjudicated plan %s for candidate %s: %s (score=%.2f)",
            request.plan_id,
            request.candidate_id,
            verdict.verdict.value,
            verdict.score,
        )
        return ModelValidationAdjudicationCompleted.model_validate(verdict)


__all__: list[str] = ["HandlerAdjudicate"]
