# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDevLaneReturnToDevEffect(NodeEffect):
    """Record dev-lane branch builds and say which are due to return to dev."""
