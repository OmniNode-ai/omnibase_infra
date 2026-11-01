# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that initiates the chain learning workflow.

Receives the inbound command and dispatches chain retrieval.
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

logger = logging.getLogger(__name__)


class HandlerChainLearnInitiate:
    """Receives chain learn command and dispatches retrieval."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        prompt_text: str,
        workflow_ref: str,
        correlation_id: UUID,
        similarity_threshold: float = 0.85,
    ) -> dict[str, str | float]:
        """Validate the inbound command and prepare retrieval dispatch.

        Args:
            prompt_text: The prompt to learn from.
            workflow_ref: The workflow that produced the chain.
            correlation_id: Workflow correlation ID.
            similarity_threshold: Minimum similarity for cache hit.

        Returns:
            Dict with dispatch parameters for the retrieval effect.
        """
        logger.info(
            "Chain learn initiated: workflow=%s, threshold=%.2f (correlation_id=%s)",
            workflow_ref,
            similarity_threshold,
            correlation_id,
        )

        return {
            "prompt_text": prompt_text,
            "workflow_ref": workflow_ref,
            "correlation_id": str(correlation_id),
            "similarity_threshold": str(similarity_threshold),
        }
