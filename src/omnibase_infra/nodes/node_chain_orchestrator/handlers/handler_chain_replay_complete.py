# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that processes replay results and dispatches verification."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainReplayResult,
)

logger = logging.getLogger(__name__)

# Minimum confidence to proceed with replay (below this -> fallback)
MIN_REPLAY_CONFIDENCE = 0.6


class HandlerChainReplayComplete:
    """Evaluates replay result and dispatches store or fallback."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        replay_result: ModelChainReplayResult,
        correlation_id: UUID,
    ) -> dict[str, str | float]:
        """Evaluate replay result and decide next step.

        Args:
            replay_result: Result from chain replay compute.
            correlation_id: Workflow correlation ID.

        Returns:
            Dict with decision: proceed to store or fallback to explore.
        """
        if replay_result.confidence >= MIN_REPLAY_CONFIDENCE:
            logger.info(
                "Replay accepted: confidence=%.2f, steps=%d (correlation_id=%s)",
                replay_result.confidence,
                len(replay_result.adapted_steps),
                correlation_id,
            )
            return {
                "action": "verify",
                "confidence": str(replay_result.confidence),
                "summary": replay_result.adaptation_summary,
            }

        logger.info(
            "Replay rejected: confidence=%.2f < %.2f (correlation_id=%s)",
            replay_result.confidence,
            MIN_REPLAY_CONFIDENCE,
            correlation_id,
        )
        return {
            "action": "fallback",
            "confidence": str(replay_result.confidence),
            "reason": f"Confidence {replay_result.confidence:.2f} below threshold {MIN_REPLAY_CONFIDENCE:.2f}",
        }
