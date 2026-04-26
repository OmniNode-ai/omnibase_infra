# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Intent effect adapter for build_loop terminal-event persistence."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.nodes.node_build_loop_projection_compute.models import (
    ModelPayloadBuildLoopAppend,
)
from omnibase_infra.nodes.node_build_loop_write_effect.handlers import (
    HandlerBuildLoopAppend,
)

logger = logging.getLogger(__name__)


class IntentEffectBuildLoopAppend:
    """Bridge `build_loop.append` intents to HandlerBuildLoopAppend."""

    def __init__(self, handler: HandlerBuildLoopAppend) -> None:
        self._handler = handler

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        typed_payload = (
            payload
            if isinstance(payload, ModelPayloadBuildLoopAppend)
            else ModelPayloadBuildLoopAppend.model_validate(payload)
        )
        await self._handler.append(typed_payload, correlation_id=correlation_id)
        logger.info(
            "build_loop append intent executed: run_id=%s correlation_id=%s",
            typed_payload.run_id,
            str(correlation_id) if correlation_id is not None else None,
        )


__all__ = ["IntentEffectBuildLoopAppend"]
