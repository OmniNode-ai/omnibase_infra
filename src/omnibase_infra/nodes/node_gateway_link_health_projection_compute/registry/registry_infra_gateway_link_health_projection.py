# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for NodeGatewayLinkHealthProjectionCompute - DI bindings and exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.nodes.node_gateway_link_health_projection_compute.node import (
    NodeGatewayLinkHealthProjectionCompute,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


class RegistryInfraGatewayLinkHealthProjection:
    """DI registry for the gateway link-health projection compute node."""

    @staticmethod
    def get_node_class() -> type[NodeGatewayLinkHealthProjectionCompute]:
        return NodeGatewayLinkHealthProjectionCompute

    @staticmethod
    def create_node(
        container: ModelONEXContainer,
    ) -> NodeGatewayLinkHealthProjectionCompute:
        return NodeGatewayLinkHealthProjectionCompute(container)


__all__ = [
    "NodeGatewayLinkHealthProjectionCompute",
    "RegistryInfraGatewayLinkHealthProjection",
]
