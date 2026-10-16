# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The result of one Bifrost contract render, with the overlay source it used (OMN-19214)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_bifrost_lane_overlay_source import (
    EnumBifrostLaneOverlaySource,
)

__all__ = ["ModelBifrostRenderOutcome"]


class ModelBifrostRenderOutcome(BaseModel):
    """What a render wrote and which lane-overlay authority supplied it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path | None = Field(
        description="The rendered contract, or None when no target is bound."
    )
    overlay_source: EnumBifrostLaneOverlaySource | None = Field(
        description="Which authority supplied the lane overlay; None when no render ran."
    )
    overlay_source_ref: str | None = Field(
        description="The store key or the overlay file path the render read."
    )
