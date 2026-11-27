# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative boot-guard effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeContractsRevisionGuardEffect(NodeEffect):
    """Refuse a boot whose mounted runtime contracts are not the image's revision."""
