# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""node_gateway_link_health_write_effect - gateway_link_health persistence node.

Declarative EFFECT node that consumes ModelPayloadGatewayLinkHealthUpsert
intents emitted by NodeGatewayLinkHealthProjectionCompute and UPSERTs them
into the public.gateway_link_health latest-known-state table.

Ticket: OMN-15570 (G3, gateway lift Phase 0)
"""

from omnibase_infra.nodes.node_gateway_link_health_write_effect.handlers import (
    HandlerGatewayLinkHealthUpsert,
)
from omnibase_infra.nodes.node_gateway_link_health_write_effect.models import (
    ModelGatewayLinkHealthUpsertResult,
)
from omnibase_infra.nodes.node_gateway_link_health_write_effect.node import (
    NodeGatewayLinkHealthWriteEffect,
)
from omnibase_infra.nodes.node_gateway_link_health_write_effect.registry import (
    RegistryInfraGatewayLinkHealthWrite,
)

__all__ = [
    "HandlerGatewayLinkHealthUpsert",
    "ModelGatewayLinkHealthUpsertResult",
    "NodeGatewayLinkHealthWriteEffect",
    "RegistryInfraGatewayLinkHealthWrite",
]
