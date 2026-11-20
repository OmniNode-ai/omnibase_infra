# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative node for retiring a declared runner pool."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeRunnerPoolRetire(NodeEffect):
    """Drain, deregister and remove a declared CI pool through the local installer."""
