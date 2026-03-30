# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative runtime node graph definition (OMN-6306).

Replaces imperative wiring with a typed, frozen model that describes
the full node graph: which nodes exist, their edges (dependencies),
and the deterministic bootstrap order.

The kernel reads this model to wire handlers, consumers, and producers
in the correct order without ad-hoc imperative code.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_node_config import ModelNodeConfig
from omnibase_infra.runtime.models.model_node_edge import ModelNodeEdge


class ModelRuntimeNodeGraph(BaseModel):
    """Declarative definition of the runtime node graph (OMN-6306).

    Provides a single frozen model that the kernel uses to determine:
      - Which nodes to instantiate (``nodes``)
      - Which dependencies exist between them (``edges``)
      - The deterministic order for bootstrapping (``bootstrap_order``)

    The ``bootstrap_order`` must list every enabled node name exactly once
    and must be topologically consistent with ``edges``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[ModelNodeConfig, ...] = Field(
        ..., description="All nodes in the runtime graph."
    )
    edges: tuple[ModelNodeEdge, ...] = Field(
        default_factory=tuple,
        description="Directed dependency edges between nodes.",
    )
    bootstrap_order: tuple[str, ...] = Field(
        ...,
        description=(
            "Deterministic startup order. Must list every enabled node "
            "name exactly once, topologically sorted by edges."
        ),
    )


__all__ = [
    "ModelRuntimeNodeGraph",
]
