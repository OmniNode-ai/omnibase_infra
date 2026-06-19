# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative coding-agent workspace COMPUTE node (OMN-13247).

Pure pre-flight workspace safety: allowed-root resolution, symlink-escape
rejection, sandbox/write-mode coherence, command-shape validation (plan §5.5).
contract.yaml + the handler drive behavior; the node body stays empty per the
declarative-node invariant.
"""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeCodingAgentWorkspaceCompute(NodeCompute):
    """Declarative pure workspace pre-flight compute node (zero custom logic)."""

    # Declarative node — all behavior defined in contract.yaml.


__all__: list[str] = ["NodeCodingAgentWorkspaceCompute"]
