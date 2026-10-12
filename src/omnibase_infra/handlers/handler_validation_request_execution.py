# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that requests execution of a freshly built validation plan.

Consumes ``validation-plan-created`` and returns the request the executor
consumes from ``validation-execution-requested``. Pure transformation: the
orchestrator emits, the executor does the I/O.

Ticket: OMN-21056
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.validation import (
    ModelValidationExecutionRequest,
    ModelValidationPlan,
)

logger = logging.getLogger(__name__)


class HandlerValidationRequestExecution:
    """Turns a created validation plan into an execution request."""

    @property
    def handler_id(self) -> str:
        """Unique identifier for this handler."""
        return "handler-validation-request-execution"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: node-level orchestration handler."""
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: pure transformation, no I/O."""
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelValidationPlan
    ) -> ModelValidationExecutionRequest:
        """Wrap the plan in the request the executor consumes.

        Args:
            request: The validation plan that was just created.

        Returns:
            The execution request carrying that plan.
        """
        logger.info(
            "Requesting execution of validation plan %s for candidate %s",
            request.plan_id,
            request.candidate_id,
        )
        return ModelValidationExecutionRequest(plan=request)


__all__: list[str] = ["HandlerValidationRequestExecution"]
