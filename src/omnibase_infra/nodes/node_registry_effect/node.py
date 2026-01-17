# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registry Effect - Declarative effect node for dual-backend registration.

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
    3. Execute infrastructure I/O via handler (Consul, PostgreSQL)
    4. Return structured response (output_model in contract)

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Declarative Handlers: handler_routing section defines dispatch rules
    - External Wiring: Handlers resolved via container dependency injection

Node Responsibilities:
    - Define I/O model contract (ModelRegistryRequest -> ModelRegistryResponse)
    - Provide dependency injection points for backend clients
    - Delegate all execution to handlers via base class

The actual handler execution and routing is performed by:
    - NodeOrchestrator (for workflow coordination)
    - Or direct handler invocation by callers

Coroutine Safety:
    This node is async-safe. Handler coordination is performed by the
    orchestrator layer, not by this effect node.

Related Modules:
    - contract.yaml: Handler routing and I/O model definitions
    - handlers/: Backend-specific handlers (Consul, PostgreSQL)
    - models/: Node-specific input/output models
    - NodeRegistrationOrchestrator: Coordinates handler execution

Migration Notes (OMN-1103):
    This declarative node replaces the imperative implementation.
    All routing logic has been moved to contract.yaml handler_routing.
    Callers should use handlers directly or via the orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient
    from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
        ProtocolPostgresAdapter,
    )


class NodeRegistryEffect(NodeEffect):
    """Declarative effect node for dual-backend node registration.

    This effect node is a lightweight shell that defines the I/O contract
    for registration operations. All routing and execution logic is driven
    by contract.yaml - this class contains no custom routing code.

    Handler coordination is performed by:
        - NodeRegistrationOrchestrator for workflow-based execution
        - Direct handler invocation for simple use cases

    Supported Operations (defined in contract.yaml handler_routing):
        - register_node: Register to both Consul and PostgreSQL
        - deregister_node: Deregister from both backends
        - retry_partial_failure: Retry a specific backend after partial failure

    Dependency Injection:
        Backend clients are injected via setter methods. The orchestrator
        or caller is responsible for wiring these dependencies.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect
        from omnibase_infra.nodes.node_registry_effect.handlers import (
            HandlerConsulRegister,
            HandlerPostgresUpsert,
        )

        # Create effect node
        container = ModelONEXContainer()
        effect = NodeRegistryEffect(container)

        # Wire backend clients
        effect.set_consul_client(consul_client)
        effect.set_postgres_adapter(postgres_adapter)

        # Execute via handlers directly
        consul_handler = HandlerConsulRegister(consul_client)
        postgres_handler = HandlerPostgresUpsert(postgres_adapter)

        # Or use orchestrator for coordinated execution
        orchestrator = NodeRegistrationOrchestrator(container)
        result = await orchestrator.execute_registration(request)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the effect node.

        Args:
            container: ONEX dependency injection container providing:
                - Backend adapters (Consul client, PostgreSQL adapter)
                - Handler instances
                - Configuration
        """
        super().__init__(container)

        # Backend adapters (injected via setter methods)
        self._consul_client: ProtocolConsulClient | None = None
        self._postgres_adapter: ProtocolPostgresAdapter | None = None

    def set_consul_client(self, client: ProtocolConsulClient) -> None:
        """Set the Consul client for service registration.

        Args:
            client: Protocol-compliant Consul client implementation.
        """
        self._consul_client = client

    def set_postgres_adapter(self, adapter: ProtocolPostgresAdapter) -> None:
        """Set the PostgreSQL adapter for registration persistence.

        Args:
            adapter: Protocol-compliant PostgreSQL adapter implementation.
        """
        self._postgres_adapter = adapter

    @property
    def consul_client(self) -> ProtocolConsulClient | None:
        """Get the Consul client if configured."""
        return self._consul_client

    @property
    def postgres_adapter(self) -> ProtocolPostgresAdapter | None:
        """Get the PostgreSQL adapter if configured."""
        return self._postgres_adapter

    @property
    def has_consul_client(self) -> bool:
        """Check if Consul client is configured."""
        return self._consul_client is not None

    @property
    def has_postgres_adapter(self) -> bool:
        """Check if PostgreSQL adapter is configured."""
        return self._postgres_adapter is not None


__all__ = ["NodeRegistryEffect"]
