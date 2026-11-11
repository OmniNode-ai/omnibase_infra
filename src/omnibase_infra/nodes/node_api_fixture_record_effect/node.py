# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative recorder; capture, masking and lifecycle belong to the handler."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeApiFixtureRecordEffect(NodeEffect):
    """Record a configured API surface without exposing bodies on the bus."""
