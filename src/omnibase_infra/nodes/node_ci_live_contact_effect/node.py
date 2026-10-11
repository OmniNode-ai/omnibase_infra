# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative CI-tooling admission effect node (OMN-18648)."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeCILiveContactEffect(NodeEffect):
    """All admission behaviour is owned by the contract-routed handler."""
