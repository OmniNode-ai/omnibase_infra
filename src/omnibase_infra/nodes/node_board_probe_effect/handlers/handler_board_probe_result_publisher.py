# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Return board-probe events through the canonical effect output path."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_board_probe_effect.models.model_board_probe_result_event import (
    ModelBoardProbeResultEvent,
)

HANDLER_ID = "board-probe-result-publisher"


class HandlerBoardProbeResultPublisher:
    """Hand one validated result event to the runtime-owned publisher."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, event: ModelBoardProbeResultEvent
    ) -> ModelHandlerOutput[None]:
        """Return the event; the contract result applier performs publication."""
        event_identity = "|".join(str(part) for part in event.key)
        return ModelHandlerOutput.for_effect(
            input_envelope_id=uuid5(
                NAMESPACE_URL, f"board-probe-input:{event_identity}"
            ),
            correlation_id=uuid5(
                NAMESPACE_URL, f"board-probe-correlation:{event.execution_id}"
            ),
            handler_id=HANDLER_ID,
            events=(event,),
        )


__all__ = ["HANDLER_ID", "HandlerBoardProbeResultPublisher"]
