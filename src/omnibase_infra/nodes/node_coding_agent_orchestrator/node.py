# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative coding-agent ORCHESTRATOR node (OMN-13247).

Sequences the workflow validate -> invoke -> capture by dispatching commands over
the bus; emits, never returns; never constructs sibling handlers in-process.
contract.yaml + the handler drive behavior; the node body stays empty per the
declarative-node invariant.
"""

from __future__ import annotations

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator


class NodeCodingAgentOrchestrator(NodeOrchestrator):
    """Declarative coding-agent workflow orchestrator (zero custom logic)."""

    # Declarative node — all behavior defined in contract.yaml.


__all__: list[str] = ["NodeCodingAgentOrchestrator"]
