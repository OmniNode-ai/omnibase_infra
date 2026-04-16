# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that initiates the scope-check workflow.

Receives ModelScopeCheckCommand and emits a ModelScopeFileReadRequest
to trigger the file read effect node.

OMN-8735: No-arg constructor required for auto-wiring compliance.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_file_read_effect.models.model_scope_file_read_request import (
    ModelScopeFileReadRequest,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.model_scope_check_command import (
    ModelScopeCheckCommand,
)

logger = logging.getLogger(__name__)


class HandlerScopeCheckInitiate:
    """Initiates scope-check workflow from an incoming command."""

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
        command: ModelScopeCheckCommand,
    ) -> ModelScopeFileReadRequest:
        """Translate scope-check command to a file read request."""
        logger.info(
            "Initiating scope-check workflow for plan_file_path=%s",
            command.plan_file_path,
        )
        return ModelScopeFileReadRequest(
            file_path=command.plan_file_path,
            correlation_id=command.correlation_id,
        )


__all__ = ["HandlerScopeCheckInitiate"]
