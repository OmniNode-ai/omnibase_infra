# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that evaluates retrieval results and dispatches replay or explore."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainRetrievalResult,
)

logger = logging.getLogger(__name__)


class HandlerChainRetrievalComplete:
    """Evaluates hit/miss from retrieval and decides next step."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        retrieval_result: ModelChainRetrievalResult,
        correlation_id: UUID,
    ) -> dict[str, str | bool]:
        """Evaluate retrieval result and decide path.

        Args:
            retrieval_result: Result from chain retrieval effect.
            correlation_id: Workflow correlation ID.

        Returns:
            Dict with path decision and data for next step.
        """
        if retrieval_result.is_hit and retrieval_result.matches:
            best_match = retrieval_result.matches[0]
            logger.info(
                "Chain HIT: similarity=%.3f, chain_id=%s (correlation_id=%s)",
                best_match.similarity_score,
                best_match.chain_entry.chain_id,
                correlation_id,
            )
            return {
                "path": "replay",
                "chain_id": str(best_match.chain_entry.chain_id),
                "is_hit": "true",
            }

        logger.info(
            "Chain MISS: best_similarity=%.3f (correlation_id=%s)",
            retrieval_result.best_match_similarity,
            correlation_id,
        )
        return {
            "path": "explore",
            "is_hit": "false",
        }
