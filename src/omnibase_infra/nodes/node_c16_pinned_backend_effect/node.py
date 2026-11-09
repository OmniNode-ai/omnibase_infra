# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative C16 pinned-backend probe effect node (OMN-23176)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeC16PinnedBackendEffect(NodeEffect):
    """All probe behaviour is owned by the contract-routed handler."""
