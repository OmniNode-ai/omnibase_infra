# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registry Effect - Declarative effect node for PostgreSQL registration.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic - all behavior from handler_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Handler Routing Pattern:
    1. Receive registration request (input_model in contract)
    2. Route to appropriate handler based on payload type (handler_routing)
    3. Execute infrastructure I/O via handler (PostgreSQL)
    4. Return structured response (output_model in contract)

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Declarative Handlers: handler_routing section defines dispatch rules
    - Container DI: Backend adapters resolved via container, not setter methods

Node Responsibilities:
    - Define I/O model contract (ModelRegistryRequest -> ModelRegistryResponse)
    - Delegate all execution to handlers via base class
    - NO custom logic - pure declarative shell

The actual handler execution and routing is performed by:
    - NodeOrchestrator (for workflow coordination)
    - Or direct handler invocation by callers

Handlers receive their dependencies directly via constructor injection:
    - HandlerPostgresUpsert(postgres_adapter)

Coroutine Safety:
    This node is async-safe. Handler coordination is performed by the
    orchestrator layer, not by this effect node.

Related Modules:
    - contract.yaml: Handler routing and I/O model definitions
    - handlers/: Backend-specific handlers (PostgreSQL)
    - models/: Node-specific input/output models
    - NodeRegistrationOrchestrator: Coordinates handler execution

Migration Notes (OMN-1726):
    Refactored from setter-based dependency injection to pure declarative shell.
    Backend adapters are now resolved via container, not via set_* methods.
    Handlers receive dependencies directly via their constructors.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


# ONEX_EXCLUDE: declarative_node - legacy effect node with direct adapter access (OMN-1725)
class NodeRegistryEffect(NodeEffect):  # ai-slop-ok: pre-existing
    """Declarative effect node for PostgreSQL-backed node registration.

    This effect node is a lightweight shell that defines the I/O contract
    for registration operations. All routing and execution logic is driven
    by contract.yaml - this class contains NO custom routing code.

    Handler coordination is performed by:
        - NodeRegistrationOrchestrator for workflow-based execution
        - Direct handler invocation for simple use cases

    Supported Operations (defined in contract.yaml handler_routing):
        - register_node: Register to PostgreSQL
        - deregister_node: Deregister from PostgreSQL
        - retry_partial_failure: Retry PostgreSQL after a failed attempt

    Dependency Injection:
        The PostgreSQL adapter is resolved via container.
        Handlers receive their dependencies directly via constructor injection.
        This node contains NO instance variables for backend clients.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect
        from omnibase_infra.nodes.node_registry_effect.handlers import (
            HandlerPostgresUpsert,
        )

        # Create effect node via container
        container = ModelONEXContainer()
        effect = NodeRegistryEffect(container)

        # Handlers receive dependencies directly via constructor
        postgres_adapter = container.resolve(ProtocolPostgresAdapter)
        postgres_handler = HandlerPostgresUpsert(postgres_adapter)

        # Or use orchestrator for coordinated execution
        orchestrator = NodeRegistrationOrchestrator(container)
        result = await orchestrator.execute_registration(request)
        ```
    """

    # Pure declarative shell - all behavior defined in contract.yaml


__all__ = ["NodeRegistryEffect"]
