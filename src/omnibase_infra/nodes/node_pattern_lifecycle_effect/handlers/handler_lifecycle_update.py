# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for applying a validation verdict to pattern lifecycle state.

Computes the tier transition by delegating to
:meth:`ModelLifecycleState.with_verdict` and returns a
:class:`ModelLifecycleResult` indicating whether the tier changed.
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumLifecycleTier,
    EnumValidationVerdict,
)
from omnibase_infra.nodes.node_pattern_lifecycle_effect.models import (
    ModelLifecycleResult,
    ModelLifecycleState,
)
from omnibase_infra.utils import sanitize_error_message

logger = logging.getLogger(__name__)


class HandlerLifecycleUpdate:
    """Apply a validation verdict to pattern lifecycle state.

    This handler is stateless.  It receives the current lifecycle state
    (or ``None`` for a first-time pattern), applies the verdict via
    ``ModelLifecycleState.with_verdict()``, and returns a result
    describing the tier transition.

    Note:
        This is an infrastructure handler (``INFRA_HANDLER``) with
        ``EFFECT`` category because it will eventually persist state
        to an external store.  In the MVP skeleton it operates purely
        in-memory.
    """

    # ------------------------------------------------------------------
    # Handler classification
    # ------------------------------------------------------------------

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-lifecycle-update"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for lifecycle writes.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            handler that manages lifecycle tier state.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting state update.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            state updates (lifecycle tier persistence).
        """
        return EnumHandlerTypeCategory.EFFECT

    # ------------------------------------------------------------------
    # Core handle method
    # ------------------------------------------------------------------

    async def handle(
        self,
        pattern_id: UUID,
        verdict: EnumValidationVerdict,
        correlation_id: UUID | None = None,
        current_state: ModelLifecycleState | None = None,
    ) -> ModelLifecycleResult:
        """Apply a validation verdict to pattern lifecycle state.

        If ``current_state`` is ``None``, a fresh state is created at the
        ``OBSERVED`` tier for the given ``pattern_id``.

        Args:
            pattern_id: Identifier of the pattern being evaluated.
            verdict: The validation verdict to apply.
            correlation_id: Correlation ID for distributed tracing.  Auto-
                generated when ``None``.
            current_state: Current lifecycle state, or None for first-time
                patterns.

        Returns:
            A ``ModelLifecycleResult`` describing the tier transition.
        """
        correlation_id = correlation_id or uuid4()
        try:
            state = current_state or ModelLifecycleState(pattern_id=pattern_id)
            previous_tier = state.current_tier

            new_state = state.with_verdict(verdict)
            new_tier = new_state.current_tier
            tier_changed = new_tier != previous_tier

            if tier_changed:
                logger.info(
                    "Pattern %s tier changed: %s -> %s (verdict=%s, cid=%s)",
                    pattern_id,
                    previous_tier.value,
                    new_tier.value,
                    verdict.value,
                    correlation_id,
                )
            else:
                logger.debug(
                    "Pattern %s tier unchanged at %s (verdict=%s, cid=%s)",
                    pattern_id,
                    new_tier.value,
                    verdict.value,
                    correlation_id,
                )

            return ModelLifecycleResult(
                pattern_id=pattern_id,
                previous_tier=previous_tier,
                new_tier=new_tier,
                tier_changed=tier_changed,
                verdict_applied=verdict,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            sanitized_error = sanitize_error_message(exc)
            logger.warning(
                "Failed to apply verdict %s to pattern %s: %s (cid=%s)",
                verdict.value,
                pattern_id,
                sanitized_error,
                correlation_id,
            )
            return ModelLifecycleResult(
                pattern_id=pattern_id,
                previous_tier=(
                    current_state.current_tier
                    if current_state is not None
                    else EnumLifecycleTier.OBSERVED
                ),
                new_tier=(
                    current_state.current_tier
                    if current_state is not None
                    else EnumLifecycleTier.OBSERVED
                ),
                tier_changed=False,
                verdict_applied=verdict,
                correlation_id=correlation_id,
                error=sanitized_error,
                error_code="LIFECYCLE_UPDATE_ERROR",
            )


__all__: list[str] = ["HandlerLifecycleUpdate"]
