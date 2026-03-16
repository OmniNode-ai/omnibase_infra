# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node Registry Effect package - Declarative effect node for dual-backend registration.

This package provides NodeRegistryEffect, a declarative effect node that coordinates
node registration against PostgreSQL.

Architecture (OMN-1103 Refactoring):
    This package follows the ONEX declarative node pattern:
    - node.py: Declarative node shell extending NodeEffect
    - models/: Node-specific Pydantic models
    - handlers/: Operation-specific handlers (PostgreSQL)
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Operation routing and I/O definitions

    The node is 100% contract-driven with zero custom business logic in node.py.
    All operation routing is defined in contract.yaml and handlers are resolved
    via container dependency injection.

Node Type: EFFECT_GENERIC
Purpose: Execute infrastructure I/O operations (PostgreSQL upsert)
         based on requests from the registration orchestrator.

Implementation Details:
    - PostgreSQL-backed registration
    - Partial failure handling with per-backend results
    - Idempotency tracking for retry safety
    - Error sanitization for security

Handlers:
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
    HandlerPartialRetry,
    HandlerPostgresDeactivate,
    HandlerPostgresUpsert,
)

# Export registry
from omnibase_infra.nodes.node_registry_effect.registry import (
    RegistryInfraRegistryEffect,
)

# Export the functional NodeRegistryEffect (direct programmatic use).
# For the ONEX declarative runtime, use node_registry_effect.node.NodeRegistryEffect.
from omnibase_infra.nodes.node_registry_effect.registry_effect import NodeRegistryEffect

__all__: list[str] = [
    # Node (functional implementation)
    "NodeRegistryEffect",
    # Registry
    "RegistryInfraRegistryEffect",
    # Handlers
    "HandlerPartialRetry",
    "HandlerPostgresDeactivate",
    "HandlerPostgresUpsert",
]
