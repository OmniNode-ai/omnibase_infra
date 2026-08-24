# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeGatewayLinkHealthWriteEffect - DI bindings and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_gateway_link_health_write_effect.node import (
    NodeGatewayLinkHealthWriteEffect,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraGatewayLinkHealthWrite:
    """DI registry for the gateway link-health write effect node."""

    @staticmethod
    def get_node_class() -> type[NodeGatewayLinkHealthWriteEffect]:
        return NodeGatewayLinkHealthWriteEffect

    @staticmethod
    def create_node(container: ModelONEXContainer) -> NodeGatewayLinkHealthWriteEffect:
        return NodeGatewayLinkHealthWriteEffect(container)


__all__ = [
    "NodeGatewayLinkHealthWriteEffect",
    "RegistryInfraGatewayLinkHealthWrite",
]
