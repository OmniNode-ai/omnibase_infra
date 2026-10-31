# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that processes store results and emits workflow completion."""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainStoreResult,
)

logger = logging.getLogger(__name__)


class HandlerChainStoreComplete:
    """Processes store result and emits completion event."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(self, request: ModelChainStoreResult) -> dict[str, str | bool]:
        """Process store result and prepare completion event.

        Args:
            request: Result from chain store effect (carries the workflow
                correlation ID).

        Returns:
            Dict with completion status.
        """
        if request.success:
            logger.info(
                "Chain stored successfully: chain_id=%s (correlation_id=%s)",
                request.chain_id,
                request.correlation_id,
            )
            return {
                "status": "complete",
                "chain_id": str(request.chain_id),
                "success": True,
            }

        logger.warning(
            "Chain store failed: %s (correlation_id=%s)",
            request.error_message,
            request.correlation_id,
        )
        return {
            "status": "complete_without_store",
            "error": request.error_message,
            "success": False,
        }
