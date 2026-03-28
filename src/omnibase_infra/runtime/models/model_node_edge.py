# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Directed dependency edge between two nodes (OMN-6306)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeEdge(BaseModel):
    """Directed dependency edge between two nodes.

    ``source`` must be bootstrapped and healthy before ``target`` starts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(..., description="Name of the upstream (dependency) node.")
    target: str = Field(..., description="Name of the downstream (dependent) node.")


__all__ = ["ModelNodeEdge"]
