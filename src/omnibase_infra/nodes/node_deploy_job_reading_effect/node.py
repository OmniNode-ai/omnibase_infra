# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative host effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDeployJobReadingEffect(NodeEffect):
    """Write a deploy job's terminal outcome as a ledger reading row."""
