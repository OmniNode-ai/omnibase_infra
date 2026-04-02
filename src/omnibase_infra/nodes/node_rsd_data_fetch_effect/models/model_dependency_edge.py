# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dependency edge model for RSD scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDependencyEdge(BaseModel):
    """A dependency edge between two tickets."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    target_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    edge_type: str = Field(default="depends_on", description="Dependency type.")
    weight: float = Field(default=1.0, description="Edge weight.")
