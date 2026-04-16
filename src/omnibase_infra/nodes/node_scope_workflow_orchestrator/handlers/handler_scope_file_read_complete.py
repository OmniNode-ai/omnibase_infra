# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for scope file read completion.

Receives ModelScopeFileReadResult and emits a ModelScopeExtractInput
to trigger the scope extraction compute node.

OMN-8735: No-arg constructor required for auto-wiring compliance.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors.error_infra import InfraConnectionError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
    ModelScopeExtractInput,
)
from omnibase_infra.nodes.node_scope_file_read_effect.models.model_scope_file_read_result import (
    ModelScopeFileReadResult,
)

logger = logging.getLogger(__name__)


class HandlerScopeFileReadComplete:
    """Processes file read result and prepares scope extraction input."""

    def __init__(self) -> None:  # stub-ok: stateless init
        """Initialize the handler (stateless)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        result: ModelScopeFileReadResult,
    ) -> ModelScopeExtractInput:
        """Translate file read result to scope extraction input."""
        logger.info(
            "Scope file read complete for file_path=%s (success=%s)",
            result.file_path,
            result.success,
        )
        if not result.success:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=result.correlation_id,
                operation="scope_file_read",
                target_name=result.file_path,
            )
            raise InfraConnectionError(
                f"Scope file read failed: {result.error_message}",
                context=context,
            )
        return ModelScopeExtractInput(
            correlation_id=result.correlation_id,
            plan_file_path=result.file_path,
            output_path=result.output_path,
            content=result.content,
        )


__all__ = ["HandlerScopeFileReadComplete"]
