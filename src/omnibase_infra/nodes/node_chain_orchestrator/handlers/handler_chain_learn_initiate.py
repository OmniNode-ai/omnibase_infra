# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that initiates the chain learning workflow.

Receives the inbound command and dispatches chain retrieval.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import ModelChainLearnCommand

logger = logging.getLogger(__name__)


class HandlerChainLearnInitiate:
    """Receives chain learn command and dispatches retrieval."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(self, request: ModelChainLearnCommand) -> dict[str, str | float]:
        """Validate the inbound command and prepare retrieval dispatch.

        Args:
            request: The chain learn command: the prompt to learn from, the
                workflow that produced the chain, the workflow correlation ID and
                the minimum similarity for a cache hit.

        Returns:
            Dict with dispatch parameters for the retrieval effect.
        """
        logger.info(
            "Chain learn initiated: workflow=%s, threshold=%.2f (correlation_id=%s)",
            request.workflow_ref,
            request.similarity_threshold,
            request.correlation_id,
        )

        return {
            "prompt_text": request.prompt_text,
            "workflow_ref": request.workflow_ref,
            "correlation_id": str(request.correlation_id),
            "similarity_threshold": str(request.similarity_threshold),
        }
