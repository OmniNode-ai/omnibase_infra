# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative runner preparation node; effects belong to its routed handler."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeployRunners(NodeEffect):
    """Converge runner TMPDIR host directories before deployment."""
