# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Effect Node v1.0.0 package.

This version provides the initial implementation of the Registry Effect Node
for dual registration to Consul and PostgreSQL via message bus bridge pattern.

Exports:
    NodeRegistryEffect: Main effect node implementation
    ModelRegistryRequest: Input model for registry operations
    ModelRegistryResponse: Output model for registry operation results
    ModelNodeIntrospectionPayload: Introspection data for node registration
    ModelConsulOperationResult: Result of a Consul operation
    ModelPostgresOperationResult: Result of a PostgreSQL operation
    ModelNodeRegistration: Node registration record from storage
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    ModelConsulOperationResult,
    ModelNodeIntrospectionPayload,
    ModelNodeRegistration,
    ModelPostgresOperationResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.node import NodeRegistryEffect

__all__: list[str] = [
    "NodeRegistryEffect",
    "ModelRegistryRequest",
    "ModelRegistryResponse",
    "ModelNodeIntrospectionPayload",
    "ModelConsulOperationResult",
    "ModelPostgresOperationResult",
    "ModelNodeRegistration",
]
