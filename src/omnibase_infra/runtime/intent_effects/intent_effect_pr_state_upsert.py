# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Intent effect adapter for pr_state persistence.

Mirrors IntentEffectBuildLoopAppend: bridges the `pr_state.upsert` intent
type (emitted by HandlerPrStateProjection.handle()) to HandlerPrStateUpsert,
so DispatchResultApplier's intent_executor can invoke the EFFECT handler
in-process rather than requiring the intent to round-trip through Kafka.
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.nodes.node_pr_state_projection_compute.models import (
    ModelPayloadPrStateUpsert,
)
from omnibase_infra.nodes.node_pr_state_write_effect.handlers import (
    HandlerPrStateUpsert,
)

logger = logging.getLogger(__name__)


class IntentEffectPrStateUpsert:
    """Bridge `pr_state.upsert` intents to HandlerPrStateUpsert."""

    def __init__(self, handler: HandlerPrStateUpsert) -> None:
        self._handler = handler

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        typed_payload = (
            payload
            if isinstance(payload, ModelPayloadPrStateUpsert)
            else ModelPayloadPrStateUpsert.model_validate(payload)
        )
        await self._handler.upsert(typed_payload, correlation_id=correlation_id)
        logger.info(
            "pr_state upsert intent executed: repo=%s pr_number=%d correlation_id=%s",
            typed_payload.repo,
            typed_payload.pr_number,
            str(correlation_id) if correlation_id is not None else None,
        )


__all__ = ["IntentEffectPrStateUpsert"]
