# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registry Effect - Declarative effect node for dual-backend registration.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom business logic - all behavior from operation_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All operation logic is 100% driven by contract.yaml, not Python code.

Operation Pattern:
    1. Receive registration request (input_model in contract)
    2. Route to appropriate handler based on operation type (operation_routing)
    3. Execute infrastructure I/O (Consul registration, PostgreSQL upsert)
    4. Return structured response (output_model in contract)

Design Decisions:
    - 100% Contract-Driven: All operation routing in YAML, not Python
    - Zero Custom Methods: Base class handles execution flow
    - Declarative Routing: Operations defined in contract's operation_routing
    - Handler Resolution: Handlers resolved via container dependency injection

Node Responsibilities:
    - Dual-backend registration (Consul + PostgreSQL)
    - Partial failure handling with per-backend results
    - Idempotency tracking for retry safety
    - Error sanitization for security

Coroutine Safety:
    This node is async-safe. Multiple operations can be processed concurrently
    when using appropriate backend adapters with connection pooling.

Related Modules:
    - contract.yaml: Operation routing and I/O model definitions
    - handlers/: Operation-specific handlers (to be implemented)
    - models/: Node-specific input/output models
    - nodes.effects.registry_effect: Legacy implementation (for reference)

Migration Notes (OMN-1103):
    This declarative node replaces the imperative implementation at
    omnibase_infra.nodes.effects.registry_effect.NodeRegistryEffect.
    The legacy implementation (507 lines) is being refactored into:
    - This declarative node shell
    - Handler classes for specific operations
    - Contract-defined operation routing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeRegistryEffect(NodeEffect):
    """Declarative effect node for dual-backend node registration.

    All behavior is defined in contract.yaml - this class contains no custom logic.
    Operations are routed to handlers via the contract's operation_routing section.

    The node coordinates registration against:
        - Consul: Service discovery registration
        - PostgreSQL: Registration record persistence

    Features:
        - Partial failure handling (one backend can fail while other succeeds)
        - Idempotency tracking for safe retries
        - Error sanitization to prevent credential exposure
        - Per-backend timing and result tracking

    Example contract.yaml operation_routing:
        ```yaml
        operation_routing:
          routing_strategy: "backend_type_match"
          operations:
            - operation_type: "consul_register"
              handler_class: "HandlerConsulRegister"
            - operation_type: "postgres_upsert"
              handler_class: "HandlerPostgresUpsert"
            - operation_type: "dual_backend_register"
              handler_class: "HandlerDualBackendRegister"
        ```

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect

        # Create via container injection
        container = ModelONEXContainer()
        effect = NodeRegistryEffect(container)

        # Operation routing defined in contract.yaml
        # Handlers resolved from container at runtime
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the effect node.

        Args:
            container: ONEX dependency injection container providing:
                - Backend adapters (Consul client, PostgreSQL adapter)
                - Idempotency store
                - Handler instances
                - Configuration
        """
        super().__init__(container)


__all__ = ["NodeRegistryEffect"]
