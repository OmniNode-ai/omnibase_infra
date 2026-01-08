# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Layer - Service integration and policy runtime.

This package provides infrastructure adapters, error handling, and the PolicyRegistry
for ONEX services. The infrastructure layer is responsible for external service
integration, transport-aware error handling, and pure decision policy management.

Core Components
---------------

**PolicyRegistry** - SINGLE SOURCE OF TRUTH for policy plugin registration:
    - Thread-safe registration of policy plugins by (policy_id, policy_type, version)
    - Enforces synchronous-by-default execution (async must be explicitly flagged)
    - Supports orchestrator and reducer policy types
    - Provides version resolution with semantic versioning
    - See: omnibase_infra.runtime.policy_registry

**Service Adapters** - External service integration:
    - PostgreSQL: Database operations
    - Kafka: Event streaming
    - Consul: Service discovery
    - Vault: Secret management
    - Valkey/Redis: Caching
    - HTTP/gRPC: API communication

**Error Handling** - Transport-aware error context:
    - ModelInfraErrorContext: Structured error metadata
    - InfraConnectionError, InfraTimeoutError, InfraAuthenticationError
    - PolicyRegistryError: Policy registration and resolution failures
    - Automatic error code selection based on transport type

**Runtime Infrastructure**:
    - RuntimeHostProcess: ONEX node execution host
    - Kernel: Contract-driven bootstrap entrypoint
    - ProtocolBindingRegistry: Handler registration
    - EventBusBindingRegistry: Event bus registration

Architecture Principles
----------------------
- **Contract-Driven**: All services follow ONEX contract patterns
- **Protocol-Based**: Duck typing through protocols, no isinstance checks
- **Strong Typing**: No Any types, Pydantic models for all data structures
- **Thread-Safe**: Registry operations protected by locks
- **Pure Policies**: Policy plugins are pure decision logic (no I/O, no side effects)

Example Usage
-------------
    >>> from omnibase_core.container import ModelONEXContainer
    >>> from omnibase_infra.runtime import PolicyRegistry
    >>> from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
    >>> from omnibase_infra.enums import EnumPolicyType
    >>>
    >>> # Container-based DI (preferred)
    >>> container = ModelONEXContainer()
    >>> await wire_infrastructure_services(container)
    >>> registry = await container.service_registry.resolve_service(PolicyRegistry)
    >>>
    >>> # Register a policy
    >>> registry.register_policy(
    ...     policy_id="exponential_backoff",
    ...     policy_class=ExponentialBackoffPolicy,
    ...     policy_type=EnumPolicyType.ORCHESTRATOR,
    ...     version="1.0.0",
    ... )
    >>>
    >>> # Retrieve and use policy
    >>> policy_cls = registry.get("exponential_backoff")
    >>> policy = policy_cls()
    >>> result = policy.evaluate(context)

See Also
--------
- PolicyRegistry: omnibase_infra.runtime.policy_registry
- Error classes: omnibase_infra.errors
- Runtime kernel: omnibase_infra.runtime.kernel
"""

from omnibase_infra import (
    clients,
    enums,
    infrastructure,
    models,
    nodes,
    shared,
    utils,
)

__all__: list[str] = [
    "clients",
    "enums",
    "infrastructure",
    "models",
    "nodes",
    "shared",
    "utils",
]
