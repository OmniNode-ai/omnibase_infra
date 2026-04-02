# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope extract compute - pure transformation node for scope extraction."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeScopeExtractCompute(NodeCompute):
    """Declarative compute node for extracting scope from plan files.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeScopeExtractCompute"]
