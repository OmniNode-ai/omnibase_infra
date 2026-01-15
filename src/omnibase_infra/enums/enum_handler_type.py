# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Type Enumeration - Architectural Role Classification.

Defines the architectural role of handlers in the ONEX system.
Each handler type drives lifecycle management, protocol selection, and runtime
invocation pattern. This is orthogonal to behavioral classification (EnumHandlerTypeCategory).

Architectural Role Semantics:
    - INFRA_HANDLER: Protocol/transport handlers that manage external connections.
        Examples: HTTP adapters, database connectors, Consul clients, Vault clients.
        Lifecycle: Connection pooling, health checks, circuit breakers.
    - NODE_HANDLER: Event processing handlers bound to ONEX nodes.
        Examples: Effect handlers, orchestrator step handlers.
        Lifecycle: Node registration, event subscription, capability advertisement.
    - PROJECTION_HANDLER: Handlers for projection read/write operations.
        Examples: State projection writers, read model updaters.
        Lifecycle: Projection versioning, consistency guarantees.
    - COMPUTE_HANDLER: Pure computation handlers with no side effects.
        Examples: Validation handlers, transformation handlers.
        Lifecycle: Stateless, idempotent, cacheable.

Design Principle:
    Handler descriptors must specify BOTH:
    1. EnumHandlerType (architectural role - this enum)
    2. EnumHandlerTypeCategory (behavioral classification)

    This separation enables:
    - Architectural role: Drives infrastructure binding and lifecycle
    - Behavioral category: Drives execution shape validation and output constraints

See Also:
    - EnumHandlerTypeCategory: Behavioral classification (EFFECT, COMPUTE, NONDETERMINISTIC_COMPUTE)
    - EnumNodeArchetype: Node archetypes for execution shape validation
    - EnumNodeOutputType: Valid node output types
    - EnumMessageCategory: Message categories for routing
    - HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md: Full handler architecture documentation
"""

from enum import Enum


class EnumHandlerType(str, Enum):
    """Handler architectural role - selects interface and lifecycle.

    These represent the architectural classification of handlers in ONEX.
    Each type drives lifecycle management, protocol selection, and runtime invocation pattern.

    Note: This is orthogonal to EnumHandlerTypeCategory which represents behavioral classification.
    Both must be specified on handler descriptors.

    Attributes:
        INFRA_HANDLER: Protocol/transport handlers (HTTP, DB, Consul, Vault).
            Manages external connections and protocol-specific operations.
            Lifecycle: Connection pooling, health checks, circuit breakers.
            Examples: DatabaseAdapter, ConsulClient, VaultClient, HttpGateway.
        NODE_HANDLER: Event processing handlers bound to nodes.
            Handles event routing and processing within the ONEX node graph.
            Lifecycle: Node registration, event subscription, capability advertisement.
            Examples: RegistrationHandler, IntrospectionHandler, RoutingHandler.
        PROJECTION_HANDLER: Projection read/write handlers.
            Manages projection state and read model operations.
            Lifecycle: Projection versioning, consistency guarantees, snapshot management.
            Examples: StateProjectionWriter, ReadModelUpdater, SnapshotHandler.
        COMPUTE_HANDLER: Pure computation handlers.
            Performs stateless, deterministic transformations.
            Lifecycle: Stateless, idempotent, cacheable - no external dependencies.
            Examples: ValidationHandler, TransformationHandler, SerializationHandler.
    """

    INFRA_HANDLER = "infra_handler"
    NODE_HANDLER = "node_handler"
    PROJECTION_HANDLER = "projection_handler"
    COMPUTE_HANDLER = "compute_handler"


__all__ = ["EnumHandlerType"]
