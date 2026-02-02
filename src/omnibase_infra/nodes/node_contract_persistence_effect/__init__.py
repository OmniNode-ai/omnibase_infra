# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Contract Persistence Effect package - Declarative effect node for contract persistence.

This package provides NodeContractPersistenceEffect, a declarative effect node that
routes intents from ContractRegistryReducer to PostgreSQL handlers for contract
and topic persistence operations.

Architecture (OMN-1845):
    This package follows the ONEX declarative node pattern:
    - node.py: Declarative node shell extending NodeEffect
    - handlers/: PostgreSQL operation handlers
    - registry/: Infrastructure registry for dependency injection
    - contract.yaml: Intent routing and I/O definitions

    The node is 100% contract-driven with zero custom business logic in node.py.
    All intent routing is defined in contract.yaml and handlers are resolved
    via container dependency injection.

Node Type: EFFECT_GENERIC
Purpose: Execute PostgreSQL I/O operations based on intents from ContractRegistryReducer.

Implementation Details:
    - Routes 6 intent types to specialized handlers
    - Circuit breaker protection for PostgreSQL
    - Error sanitization for security
    - Retry policies for transient failures

Supported Intent Types:
    - postgres.upsert_contract: Insert/update contract record
    - postgres.update_topic: Update topic routing table
    - postgres.mark_stale: Batch mark stale contracts
    - postgres.update_heartbeat: Update heartbeat timestamp
    - postgres.deactivate_contract: Soft delete contract
    - postgres.cleanup_topic_references: Remove contract from topics

Handlers:
    - HandlerPostgresContractUpsert: Contract upsert operations
    - HandlerPostgresTopicUpdate: Topic routing updates
    - HandlerPostgresMarkStale: Batch staleness marking
    - HandlerPostgresHeartbeat: Heartbeat timestamp updates
    - HandlerPostgresDeactivate: Contract deactivation
    - HandlerPostgresCleanupTopics: Topic reference cleanup

Usage:
    ```python
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.node_contract_persistence_effect import (
        NodeContractPersistenceEffect,
    )

    # Create via container injection
    container = ModelONEXContainer()
    effect = NodeContractPersistenceEffect(container)
    ```

Related:
    - contract.yaml: Intent routing definition
    - node.py: Declarative node implementation
    - handlers/: PostgreSQL operation handlers
    - registry/: Infrastructure registry
    - contract_registry_reducer/: Source of intents
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket
"""

from __future__ import annotations

# Export handlers
from omnibase_infra.nodes.node_contract_persistence_effect.handlers import (
    HandlerPostgresCleanupTopics,
    HandlerPostgresContractUpsert,
    HandlerPostgresDeactivate,
    HandlerPostgresHeartbeat,
    HandlerPostgresMarkStale,
    HandlerPostgresTopicUpdate,
)

# Export the declarative node
from omnibase_infra.nodes.node_contract_persistence_effect.node import (
    NodeContractPersistenceEffect,
)

# Export registry
from omnibase_infra.nodes.node_contract_persistence_effect.registry import (
    RegistryInfraContractPersistenceEffect,
)

__all__: list[str] = [
    # Node
    "NodeContractPersistenceEffect",
    # Registry
    "RegistryInfraContractPersistenceEffect",
    # Handlers
    "HandlerPostgresCleanupTopics",
    "HandlerPostgresContractUpsert",
    "HandlerPostgresDeactivate",
    "HandlerPostgresHeartbeat",
    "HandlerPostgresMarkStale",
    "HandlerPostgresTopicUpdate",
]
