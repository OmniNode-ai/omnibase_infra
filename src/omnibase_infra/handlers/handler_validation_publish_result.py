# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that publishes the final validation result.

Consumes ``validation-adjudication-completed`` and returns the result published
on ``validation-result-published``. Pure transformation: the verdict is the
adjudicator's; the orchestrator publishes it as the pipeline's terminal event.

Ticket: OMN-21056
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation.model_validation_result_published import (
    ModelValidationResultPublished,
)
from omnibase_infra.models.validation.model_verdict import (
    ModelVerdict,
)

logger = logging.getLogger(__name__)


class HandlerValidationPublishResult:
    """Turns the adjudicator's verdict into the published validation result."""

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-validation-publish-result"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level orchestration handler."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure transformation, no I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(self, request: ModelVerdict) -> ModelValidationResultPublished:
        """Republish the verdict as the pipeline's final result.

        Args:
            request: The verdict the adjudicator produced.

        Returns:
            The same verdict as the published validation result.
        """
        logger.info(
            "Publishing validation result %s for candidate %s (plan %s)",
            request.verdict.value,
            request.candidate_id,
            request.plan_id,
        )
        return ModelValidationResultPublished.model_validate(
            request, from_attributes=True
        )


__all__: list[str] = ["HandlerValidationPublishResult"]
