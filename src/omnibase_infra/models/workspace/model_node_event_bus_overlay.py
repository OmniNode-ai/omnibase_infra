# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``event_bus`` block of a node's contract overlay (OMN-22362)."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType


class ModelNodeEventBusOverlay(BaseModel):
    """``event_bus`` of ``<config root>/config/onex/overlays/<node>/overlay.yaml``.

    ``type: kafka`` (the lane's Redpanda) names the lane whose declared broker the client
    dials; the broker address and transport stay in the lane declaration. ``type: inmemory``
    hosts the node in the caller's process and names no lane. The shape omnimarket's
    ``onex scm`` reads from the same overlay (OMN-22351).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: EnumEventBusType
    lane: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9-]*$", max_length=64
    )

    @model_validator(mode="after")
    def _lane_matches_type(self) -> Self:
        if self.type == EnumEventBusType.KAFKA and self.lane is None:
            raise ValueError("event_bus.type kafka must name the lane it dials")
        if self.type == EnumEventBusType.INMEMORY and self.lane is not None:
            raise ValueError(
                "event_bus.type inmemory has no broker, so it names no lane"
            )
        if self.type == EnumEventBusType.CLOUD:
            raise ValueError("event_bus.type must be kafka or inmemory")
        return self


__all__: list[str] = ["ModelNodeEventBusOverlay"]
