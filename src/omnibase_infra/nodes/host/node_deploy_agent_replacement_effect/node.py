# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative deploy-agent replacement effect (OMN-19998)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeployAgentReplacementEffect(NodeEffect):
    """The contract-routed handler owns admission and recreate supervision."""
