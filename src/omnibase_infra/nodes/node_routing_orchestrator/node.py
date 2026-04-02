# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routing orchestrator - coordinates the model routing workflow."""

from __future__ import annotations

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator


class NodeRoutingOrchestrator(NodeOrchestrator):
    """Declarative orchestrator node for model routing workflow.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRoutingOrchestrator"]
