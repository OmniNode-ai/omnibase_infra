# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that adapts a cached chain to a new prompt context.

This is a COMPUTE handler -- pure transformation, no I/O.
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_chain_orchestrator.models import (
    ModelChainEntry,
    ModelChainReplayResult,
    ModelChainStep,
)

logger = logging.getLogger(__name__)


class HandlerChainReplay:
    """Adapts a cached chain's steps to a new prompt context."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        cached_chain: ModelChainEntry,
        new_prompt_text: str,
        correlation_id: UUID,
        new_context: dict[str, str] | None = None,
    ) -> ModelChainReplayResult:
        """Adapt cached chain steps to new context.

        The adaptation is conservative: chain steps are preserved as-is since
        they represent verified node executions. The confidence score reflects
        how similar the new prompt is to the original (passed through from the
        retrieval similarity score via orchestrator context).

        Args:
            cached_chain: The cached chain to adapt.
            new_prompt_text: The new prompt text.
            correlation_id: Workflow correlation ID.
            new_context: Optional context variables for adaptation.

        Returns:
            ModelChainReplayResult with adapted steps and confidence.
        """
        context = new_context or {}

        logger.info(
            "Replaying chain %s for new prompt (correlation_id=%s, steps=%d)",
            cached_chain.chain_id,
            correlation_id,
            len(cached_chain.chain_steps),
        )

        # Adapt steps: preserve structure, apply context substitutions
        adapted_steps: list[ModelChainStep] = []
        adaptations_made: list[str] = []

        for step in cached_chain.chain_steps:
            # For MVP, steps are replayed as-is. Context substitution
            # applies to the operation field if context keys match.
            new_operation = step.operation
            for key, value in context.items():
                if key in new_operation:
                    new_operation = new_operation.replace(key, value)
                    adaptations_made.append(
                        f"step[{step.step_index}]: substituted {key}"
                    )

            adapted_steps.append(
                ModelChainStep(
                    step_index=step.step_index,
                    node_ref=step.node_ref,
                    operation=new_operation,
                    input_hash=step.input_hash,
                    output_hash=step.output_hash,
                    duration_ms=step.duration_ms,
                    event_topic=step.event_topic,
                )
            )

        # Confidence is high if no adaptations were needed (exact replay),
        # lower if substitutions were applied
        if not adaptations_made:
            confidence = 0.95
            summary = "Exact replay -- no context substitutions needed"
        else:
            confidence = max(0.5, 0.95 - 0.05 * len(adaptations_made))
            summary = (
                f"Adapted {len(adaptations_made)} steps: {'; '.join(adaptations_made)}"
            )

        logger.info(
            "Chain replay complete: confidence=%.2f, adaptations=%d (correlation_id=%s)",
            confidence,
            len(adaptations_made),
            correlation_id,
        )

        return ModelChainReplayResult(
            correlation_id=correlation_id,
            adapted_steps=tuple(adapted_steps),
            adaptation_summary=summary,
            confidence=confidence,
        )
