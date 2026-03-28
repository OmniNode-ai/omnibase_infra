# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration for a single node in the runtime graph (OMN-6306)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeConfig(BaseModel):
    """Configuration for a single node in the runtime graph.

    Each node declares its identity, handler type, and the topics
    it subscribes to and publishes on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Unique node identifier within the graph.")
    handler_class: str = Field(
        ...,
        description=(
            "Fully qualified Python class path for the handler "
            "(e.g. 'omnibase_infra.nodes...HandlerFoo')."
        ),
    )
    subscribe_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topic suffixes this node consumes from.",
    )
    publish_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topic suffixes this node publishes to.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the node is active in this graph configuration.",
    )


__all__ = ["ModelNodeConfig"]
