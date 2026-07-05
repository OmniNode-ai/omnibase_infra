# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet health compute -- pure per-runner + fleet-level classification (OMN-13942)."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeRunnerFleetHealthCompute(NodeCompute):
    """Declarative compute node for classifying a runner-fleet snapshot.

    All behavior is defined in contract.yaml - no custom logic here.
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRunnerFleetHealthCompute"]
