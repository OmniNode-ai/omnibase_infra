# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routing score reducer - tracks model capability scores over time."""

from __future__ import annotations

from omnibase_core.nodes.node_reducer import NodeReducer


class NodeRoutingScoreReducer(NodeReducer):
    """Declarative reducer node for tracking model capability scores.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRoutingScoreReducer"]
