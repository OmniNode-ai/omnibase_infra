# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registry Effect package - Declarative effect node for dual-backend registration.

This package provides NodeRegistryEffect, a declarative effect node that coordinates
node registration against both Consul (service discovery) and PostgreSQL (persistence).

Architecture (OMN-1103 Refactoring):
    This package follows the ONEX declarative node pattern:
    - node.py: Declarative node shell extending NodeEffect
    - models/: Node-specific Pydantic models
    - handlers/: Operation-specific handlers (PostgreSQL, Consul)
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Operation routing and I/O definitions

    The node is 100% contract-driven with zero custom business logic in node.py.
    All operation routing is defined in contract.yaml and handlers are resolved
    via container dependency injection.

Node Type: EFFECT_GENERIC
Purpose: Execute infrastructure I/O operations (Consul registration, PostgreSQL upsert)
         based on requests from the registration orchestrator.

Implementation Details:
    - Dual-backend registration (Consul + PostgreSQL)
    - Partial failure handling with per-backend results
    - Idempotency tracking for retry safety
    - Error sanitization for security

Handlers:
    - HandlerConsulRegister: Consul service registration
    - HandlerConsulDeregister: Consul service deregistration
    - HandlerPostgresUpsert: PostgreSQL registration record upsert
    - HandlerPostgresDeactivate: PostgreSQL registration deactivation
    - HandlerPartialRetry: Targeted retry for partial failures

Usage:
    ```python
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect

    # Create via container injection
    container = ModelONEXContainer()
    effect = NodeRegistryEffect(container)
    ```

Related:
    - contract.yaml: Operation routing definition
    - node.py: Declarative node implementation
    - models/: Node-specific models
    - handlers/: Operation handlers
    - registry/: Infrastructure registry
"""

from __future__ import annotations

# Export handlers
from omnibase_infra.nodes.node_registry_effect.handlers import (
    HandlerConsulDeregister,
    HandlerConsulRegister,
    HandlerPartialRetry,
    HandlerPostgresDeactivate,
    HandlerPostgresUpsert,
)

# Export the declarative node
from omnibase_infra.nodes.node_registry_effect.node import NodeRegistryEffect

# Export registry
from omnibase_infra.nodes.node_registry_effect.registry import (
    RegistryInfraRegistryEffect,
)

__all__: list[str] = [
    # Node
    "NodeRegistryEffect",
    # Registry
    "RegistryInfraRegistryEffect",
    # Handlers
    "HandlerConsulDeregister",
    "HandlerConsulRegister",
    "HandlerPartialRetry",
    "HandlerPostgresDeactivate",
    "HandlerPostgresUpsert",
]
