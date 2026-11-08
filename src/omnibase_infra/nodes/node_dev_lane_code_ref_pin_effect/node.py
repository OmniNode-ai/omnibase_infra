# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDevLaneCodeRefPinEffect(NodeEffect):
    """Register, read and release the pin that keeps the dev deploy agent on a branch."""
