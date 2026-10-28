# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-node serving counts and their health verdict (OMN-21421)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.placement.enum_placement_health_verdict import (
    EnumPlacementHealthVerdict,
)
from omnibase_infra.handlers.placement.model_placement_serving_count import (
    ModelPlacementServingCount,
)


class ModelPlacementServingReport(BaseModel):
    """The serving count of every expected node and the health verdict over them."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    counts: tuple[ModelPlacementServingCount, ...] = ()
    min_serving: int = Field(default=2, ge=1)
    verdict: EnumPlacementHealthVerdict
    under_served: tuple[str, ...] = ()


__all__ = ["ModelPlacementServingReport"]
