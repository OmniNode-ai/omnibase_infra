# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for one placement decision: which nodes are served by how many runtimes (OMN-22132)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPlacementServingRequest(BaseModel):
    """Ask the placement runtime for its serving report."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    min_serving: int | None = Field(
        default=None,
        gt=0,
        description="Override of the contract profile's min_serving; None keeps it.",
    )


__all__ = ["ModelPlacementServingRequest"]
