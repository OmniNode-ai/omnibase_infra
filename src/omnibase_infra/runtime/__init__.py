# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime module for omnibase_infra.

This module provides the runtime infrastructure for the ONEX infrastructure layer,
including three SINGLE SOURCE OF TRUTH registries and the runtime execution host.

Core Registries
---------------
- **PolicyRegistry**: SINGLE SOURCE OF TRUTH for policy plugin registration
    - Container-based dependency injection support (preferred) or singleton accessor (legacy)
    - Thread-safe registration by (policy_id, policy_type, version)
    - Enforces synchronous-by-default execution (async must be explicit)
    - Supports orchestrator and reducer policy types with version resolution
    - Pure decision logic plugins (no I/O, no side effects)
    - Integrates with ModelOnexContainer for DI pattern

- **ProtocolBindingRegistry**: SINGLE SOURCE OF TRUTH for protocol handler registration
    - Maps handler types to handler implementations
    - Enables protocol-based dependency injection
    - Supports HTTP, database, Kafka, Vault, Consul, Valkey/Redis, gRPC handlers

- **EventBusBindingRegistry**: Registry for event bus implementations
    - Maps event bus kinds to event bus implementations
    - Supports in-memory and Kafka event buses
    - Enables event-driven architectures

Runtime Components
------------------
- **Kernel**: Contract-driven bootstrap entrypoint for the ONEX runtime
- **RuntimeHostProcess**: Infrastructure-specific runtime host process implementation
- **ServiceHealth**: HTTP health check endpoint for container orchestration
- **Wiring functions**: Register handlers and event buses with registries
- **Envelope validation**: Validate event envelope structures

Message Dispatch Engine
-----------------------
- **MessageDispatchEngine**: Runtime dispatch engine for message routing
- **DispatcherRegistry**: Thread-safe registry for message dispatchers with freeze pattern
- **ProtocolMessageDispatcher**: Protocol for category-based message dispatchers

Chain-Aware Dispatch (OMN-951)
------------------------------
- **ChainAwareDispatcher**: Dispatch wrapper with correlation/causation chain validation
- **propagate_chain_context**: Helper to propagate chain context from parent to child
- **validate_dispatch_chain**: Validate chain propagation and raise on violations

The runtime module serves as the entry point for running infrastructure services
and configuring the handler and policy ecosystem.
"""

from __future__ import annotations

# isort: off
# NOTE: Import order matters here to avoid circular import in omnibase_core.
# The chain_aware_dispatch module imports ModelEventEnvelope which triggers complex
# import chains in omnibase_core. By importing message_dispatch_engine first via
# DispatchContextEnforcer, we warm the sys.modules cache before chain_aware_dispatch.

from omnibase_infra.runtime.dispatch_context_enforcer import DispatchContextEnforcer
from omnibase_infra.runtime.registry_dispatcher import (
    DispatcherRegistry,
    ProtocolMessageDispatcher,
)
from omnibase_infra.runtime.envelope_validator import (
    PAYLOAD_REQUIRED_OPERATIONS,
    validate_envelope,
)
from omnibase_infra.runtime.handler_registry import (
    EVENT_BUS_INMEMORY,
    EVENT_BUS_KAFKA,
    HANDLER_TYPE_CONSUL,
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_GRPC,
    HANDLER_TYPE_HTTP,
    HANDLER_TYPE_KAFKA,
    HANDLER_TYPE_VALKEY,
    HANDLER_TYPE_VAULT,
    EventBusBindingRegistry,
    ProtocolBindingRegistry,
    RegistryError,
    get_event_bus_class,
    get_event_bus_registry,
    get_handler_class,
    get_handler_registry,
    register_handlers_from_config,
)

# ServiceHealth moved to services/ directory (OMN-529)
# Import from omnibase_infra.services.service_health instead
from omnibase_infra.runtime.kernel import bootstrap as kernel_bootstrap
from omnibase_infra.runtime.kernel import load_runtime_config
from omnibase_infra.runtime.kernel import main as kernel_main
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.models import (
    ModelRuntimeSchedulerConfig,
    ModelRuntimeSchedulerMetrics,
    ModelRuntimeTick,
)
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.protocol_policy import ProtocolPolicy
from omnibase_infra.runtime.protocols import ProtocolRuntimeScheduler
from omnibase_infra.runtime.registry import (
    MessageTypeRegistry,
    MessageTypeRegistryError,
    ModelDomainConstraint,
    ModelMessageTypeEntry,
    ProtocolMessageTypeRegistry,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.runtime_scheduler import RuntimeScheduler
from omnibase_infra.runtime.wiring import (
    get_known_event_bus_kinds,
    get_known_handler_types,
    wire_custom_event_bus,
    wire_custom_handler,
    wire_default_handlers,
    wire_handlers_from_contract,
)

# Container wiring (OMN-888)
from omnibase_infra.runtime.container_wiring import (
    get_compute_registry_from_container,
    get_handler_node_introspected_from_container,
    get_handler_node_registration_acked_from_container,
    get_handler_registry_from_container,
    get_handler_runtime_tick_from_container,
    get_or_create_compute_registry,
    get_or_create_policy_registry,
    get_policy_registry_from_container,
    get_projection_reader_from_container,
    wire_infrastructure_services,
    wire_registration_dispatchers,
    wire_registration_handlers,
)

# Registration dispatchers (OMN-892)
from omnibase_infra.runtime.dispatchers import (
    DispatcherNodeIntrospected,
    DispatcherNodeRegistrationAcked,
    DispatcherRuntimeTick,
)

# Introspection event router (PR #101)
from omnibase_infra.runtime.introspection_event_router import (
    IntrospectionEventRouter,
)

# Chain-aware dispatch (OMN-951) - must be imported LAST to avoid circular import
from omnibase_infra.runtime.chain_aware_dispatch import (
    ChainAwareDispatcher,
    propagate_chain_context,
    validate_dispatch_chain,
)

# isort: on

# =============================================================================
# DEPRECATION ALIASES (OMN-529)
# These symbols were moved to omnibase_infra.services.service_health
# Re-exported here for backward compatibility - will be removed in v0.5.0
# =============================================================================
_DEPRECATED_SYMBOLS: dict[str, str] = {
    "ServiceHealth": "omnibase_infra.services.service_health",
    "DEFAULT_HTTP_HOST": "omnibase_infra.services.service_health",
    "DEFAULT_HTTP_PORT": "omnibase_infra.services.service_health",
}


def __getattr__(name: str) -> object:
    """Lazy-load deprecated symbols with deprecation warnings.

    This function is called when an attribute is not found in the module.
    It handles deprecated re-exports by:
    1. Checking if the name is a deprecated symbol
    2. Issuing a DeprecationWarning
    3. Returning the actual symbol from its new location

    Args:
        name: The attribute name being accessed.

    Returns:
        The requested symbol from its new location.

    Raises:
        AttributeError: If the name is not a deprecated symbol.
    """
    import warnings

    if name in _DEPRECATED_SYMBOLS:
        new_module = _DEPRECATED_SYMBOLS[name]
        warnings.warn(
            f"'{name}' is deprecated in omnibase_infra.runtime and will be removed "
            f"in v0.5.0. Import from '{new_module}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from the new location
        from omnibase_infra.services.service_health import (
            DEFAULT_HTTP_HOST as _DEFAULT_HTTP_HOST,
        )
        from omnibase_infra.services.service_health import (
            DEFAULT_HTTP_PORT as _DEFAULT_HTTP_PORT,
        )
        from omnibase_infra.services.service_health import (
            ServiceHealth as _ServiceHealth,
        )

        _symbol_map: dict[str, object] = {
            "ServiceHealth": _ServiceHealth,
            "DEFAULT_HTTP_HOST": _DEFAULT_HTTP_HOST,
            "DEFAULT_HTTP_PORT": _DEFAULT_HTTP_PORT,
        }
        return _symbol_map[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__: list[str] = [
    # Deprecated re-exports (OMN-529) - will be removed in v0.5.0
    # Import from omnibase_infra.services.service_health instead
    "DEFAULT_HTTP_HOST",
    "DEFAULT_HTTP_PORT",
    "ServiceHealth",
    # Event bus kind constants
    "EVENT_BUS_INMEMORY",
    "EVENT_BUS_KAFKA",
    "HANDLER_TYPE_CONSUL",
    "HANDLER_TYPE_DATABASE",
    "HANDLER_TYPE_GRPC",
    # Handler type constants
    "HANDLER_TYPE_HTTP",
    "HANDLER_TYPE_KAFKA",
    "HANDLER_TYPE_VALKEY",
    "HANDLER_TYPE_VAULT",
    # Envelope validation
    "PAYLOAD_REQUIRED_OPERATIONS",
    # Chain-aware dispatch (OMN-951)
    "ChainAwareDispatcher",
    # Context enforcement
    "DispatchContextEnforcer",
    "DispatcherRegistry",
    "EventBusBindingRegistry",
    # Message dispatch engine
    "MessageDispatchEngine",
    # Message type registry (OMN-937)
    "MessageTypeRegistry",
    "MessageTypeRegistryError",
    "ModelDomainConstraint",
    "ModelMessageTypeEntry",
    "ModelRuntimeSchedulerConfig",
    "ModelRuntimeSchedulerMetrics",
    "ModelRuntimeTick",
    "PolicyRegistry",
    # Registry classes
    "ProtocolBindingRegistry",
    "ProtocolMessageDispatcher",
    "ProtocolMessageTypeRegistry",
    # Policy protocol and registry
    "ProtocolPolicy",
    # Runtime scheduler (OMN-953)
    "ProtocolRuntimeScheduler",
    # Error class
    "RegistryError",
    # Runtime host
    "RuntimeHostProcess",
    "RuntimeScheduler",
    "get_compute_registry_from_container",
    "get_event_bus_class",
    "get_event_bus_registry",
    # Convenience functions
    "get_handler_class",
    "get_handler_node_introspected_from_container",
    "get_handler_node_registration_acked_from_container",
    # Singleton accessors
    "get_handler_registry",
    "get_handler_registry_from_container",
    "get_handler_runtime_tick_from_container",
    "get_known_event_bus_kinds",
    "get_known_handler_types",
    "get_or_create_compute_registry",
    "get_or_create_policy_registry",
    "get_policy_registry_from_container",
    "get_projection_reader_from_container",
    # Kernel entrypoint
    "kernel_bootstrap",
    "kernel_main",
    "load_runtime_config",
    "propagate_chain_context",
    "register_handlers_from_config",
    "validate_dispatch_chain",
    "validate_envelope",
    "wire_custom_event_bus",
    "wire_custom_handler",
    # Wiring functions
    "wire_default_handlers",
    "wire_handlers_from_contract",
    # Container wiring (OMN-888)
    "wire_infrastructure_services",
    "wire_registration_handlers",
    "wire_registration_dispatchers",
    # Registration dispatchers (OMN-892)
    "DispatcherNodeIntrospected",
    "DispatcherRuntimeTick",
    "DispatcherNodeRegistrationAcked",
    # Introspection event router (PR #101)
    "IntrospectionEventRouter",
]
