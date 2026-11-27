# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative boot-guard effect; the handler owns all operations."""

from omnibase_core.nodes.node_effect import NodeEffect


class NodeCloudLabGuardEffect(NodeEffect):
    """Refuse a cloud-mode boot while a setting holds a lab address."""
