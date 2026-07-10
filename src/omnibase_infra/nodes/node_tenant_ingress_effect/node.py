# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tenant-ingress effect node.

The long-running ServiceTenantIngress owns publishing; the handler under
this node only transforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeTenantIngressEffect(NodeEffect):
    """Contract-driven effect node for the tenant-ingress trust boundary."""

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize with container dependency injection."""
        super().__init__(container)


__all__ = ["NodeTenantIngressEffect"]
