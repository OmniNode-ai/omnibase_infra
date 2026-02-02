# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Contract Persistence Effect - Declarative effect node for contract registry persistence.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic - all behavior from handler_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Handler Routing Pattern:
    1. Receive intent from ContractRegistryReducer (ModelIntent with typed payload)
    2. Route to appropriate handler based on payload.intent_type (handler_routing)
    3. Execute PostgreSQL I/O via handler
    4. Return structured response (output_model in contract)

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Declarative Handlers: handler_routing section defines dispatch rules
    - Container DI: Backend adapters resolved via container, not setter methods

Supported Intent Types (from ContractRegistryReducer):
    - postgres.upsert_contract: Insert/update contract record
    - postgres.update_topic: Update topic routing table
    - postgres.mark_stale: Batch mark contracts as stale
    - postgres.update_heartbeat: Update last_seen_at timestamp
    - postgres.deactivate_contract: Mark contract as inactive (soft delete)
    - postgres.cleanup_topic_references: Remove contract from topic arrays

Node Responsibilities:
    - Route intents to appropriate PostgreSQL handlers
    - Delegate all execution to handlers via base class
    - NO custom logic - pure declarative shell

Related Modules:
    - contract.yaml: Handler routing and I/O model definitions
    - handlers/: PostgreSQL operation handlers
    - contract_registry_reducer/: Source of intents
    - models/model_payload_*.py: Intent payload types

Related Tickets:
    - OMN-1845: NodeContractPersistenceEffect implementation
    - OMN-1653: ContractRegistryReducer (source of intents)
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeContractPersistenceEffect(NodeEffect):
    """Declarative effect node for contract registry persistence.

    This effect node is a lightweight shell that routes intents from
    ContractRegistryReducer to PostgreSQL handlers. All routing and
    execution logic is driven by contract.yaml - this class contains
    NO custom routing code.

    Supported Intent Types:
        - postgres.upsert_contract: Upsert contract record
        - postgres.update_topic: Update topic routing table
        - postgres.mark_stale: Batch mark stale contracts
        - postgres.update_heartbeat: Update heartbeat timestamp
        - postgres.deactivate_contract: Soft delete contract
        - postgres.cleanup_topic_references: Remove contract from topics

    Dependency Injection:
        Backend adapters (PostgreSQL) are resolved via container.
        Handlers receive their dependencies directly via constructor injection.
        This node contains NO instance variables for backend clients.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_contract_persistence_effect import (
            NodeContractPersistenceEffect,
        )

        # Create effect node via container
        container = ModelONEXContainer()
        effect = NodeContractPersistenceEffect(container)

        # Handlers receive dependencies directly via constructor
        postgres_adapter = container.resolve(ProtocolPostgresAdapter)
        upsert_handler = HandlerPostgresContractUpsert(postgres_adapter)

        # Execute handler with intent payload
        result = await upsert_handler.handle(intent_payload)
        ```
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeContractPersistenceEffect"]
