# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Node shell for waste detection compute."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeWasteDetectionCompute(NodeCompute):
    """Compute node for session-windowed LLM waste detection."""


__all__ = ["NodeWasteDetectionCompute"]
