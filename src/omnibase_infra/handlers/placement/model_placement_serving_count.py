# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How many live runtimes load one node (OMN-21421)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPlacementServingCount(BaseModel):
    """How many live runtimes load one node, and which."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_key: str = Field(min_length=1)
    serving: int = Field(ge=0)
    runtime_ids: tuple[str, ...] = ()


__all__ = ["ModelPlacementServingCount"]
