# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative effect node; the handler owns the three event publications."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeDevRollEventsEffect(NodeEffect):
    """Publish the dev roll's start, manifest and incident events."""
