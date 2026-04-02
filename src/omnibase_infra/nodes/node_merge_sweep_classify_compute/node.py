# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Merge-sweep classify compute - pure PR classification logic."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeMergeSweepClassifyCompute(NodeCompute):
    """Declarative compute node for classifying PRs into Track A/B.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeMergeSweepClassifyCompute"]
