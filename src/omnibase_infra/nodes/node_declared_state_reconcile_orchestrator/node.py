# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative orchestrator node for declared-state reconcile rounds."""

from omnibase_core.nodes.node_orchestrator import NodeOrchestrator


class NodeDeclaredStateReconcileOrchestrator(NodeOrchestrator):
    """Run observe, diff and apply rounds through the contract's handler."""
