# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""node_gateway_link_health_projection_compute - gateway link-health projection node.

Declarative COMPUTE node that subscribes to
onex.evt.omnibase-infra.gateway-heartbeat.v1 (published by
node_bus_forwarder_effect) and emits ModelIntent payloads for the EFFECT
layer to persist into gateway_link_health.

Ticket: OMN-15570 (G3, gateway lift Phase 0)
"""

from omnibase_infra.nodes.node_gateway_link_health_projection_compute.handlers import (
    HandlerGatewayLinkHealthProjection,
)
from omnibase_infra.nodes.node_gateway_link_health_projection_compute.models import (
    ModelPayloadGatewayLinkHealthUpsert,
)
from omnibase_infra.nodes.node_gateway_link_health_projection_compute.node import (
    NodeGatewayLinkHealthProjectionCompute,
)
from omnibase_infra.nodes.node_gateway_link_health_projection_compute.registry import (
    RegistryInfraGatewayLinkHealthProjection,
)

__all__ = [
    "HandlerGatewayLinkHealthProjection",
    "ModelPayloadGatewayLinkHealthUpsert",
    "NodeGatewayLinkHealthProjectionCompute",
    "RegistryInfraGatewayLinkHealthProjection",
]
