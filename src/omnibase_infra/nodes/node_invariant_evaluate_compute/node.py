# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative compute node for invariant evaluation."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeInvariantEvaluateCompute(NodeCompute):
    """Declarative shell; behavior is routed by contract.yaml."""


__all__ = ["NodeInvariantEvaluateCompute"]
