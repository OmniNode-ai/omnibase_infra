# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What was applied to one topic's consumer before it subscribed (OMN-21753)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.event_bus.model_offset_handover_result import (
    ModelOffsetHandoverResult,
)


class ModelContractSubscribePreflightResult(BaseModel):
    """The declarations ``HandlerContractSubscribePreflight`` applied for a topic.

    Attributes:
        topic: Canonical topic the consumer subscribes to.
        max_in_flight_records: The declared consume bound; 1 is the unchanged
            inline serial path and is not declared to the bus.
        handover: The consumer-group handover, ``None`` when the contract
            declares none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1)
    max_in_flight_records: int = Field(..., ge=1)
    handover: ModelOffsetHandoverResult | None = None
