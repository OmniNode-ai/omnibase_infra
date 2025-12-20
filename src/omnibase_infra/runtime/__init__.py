# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime module for omnibase_infra.

Migration Note (OMN-934):
    This module renames "Handler" terminology to "Dispatcher" for message routing.
    Legacy aliases are provided for backwards compatibility in MessageDispatchEngine:

    - register_handler() -> register_dispatcher() (method alias provided)
    - handler_count -> dispatcher_count (property alias provided)
    - get_handler_metrics() -> get_dispatcher_metrics() (method alias provided)

    Note: DispatcherRegistry is a NEW class (no HandlerRegistry predecessor existed).

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
- **HealthServer**: HTTP health check endpoint for container orchestration
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
from omnibase_infra.runtime.dispatcher_registry import (
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
from omnibase_infra.runtime.health_server import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    HealthServer,
)
from omnibase_infra.runtime.kernel import bootstrap as kernel_bootstrap
from omnibase_infra.runtime.kernel import load_runtime_config
from omnibase_infra.runtime.kernel import main as kernel_main
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.protocol_policy import ProtocolPolicy
from omnibase_infra.runtime.registry import (
    MessageTypeRegistry,
    MessageTypeRegistryError,
    ModelDomainConstraint,
    ModelMessageTypeEntry,
    ProtocolMessageTypeRegistry,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.wiring import (
    get_known_event_bus_kinds,
    get_known_handler_types,
    wire_custom_event_bus,
    wire_custom_handler,
    wire_default_handlers,
    wire_handlers_from_contract,
)

# Chain-aware dispatch (OMN-951) - must be imported LAST to avoid circular import
from omnibase_infra.runtime.chain_aware_dispatch import (
    ChainAwareDispatcher,
    propagate_chain_context,
    validate_dispatch_chain,
)

# isort: on

__all__: list[str] = [
    # Kernel entrypoint
    "kernel_bootstrap",
    "kernel_main",
    "load_runtime_config",
    # Runtime host
    "RuntimeHostProcess",
    # Health server
    "HealthServer",
    "DEFAULT_HTTP_PORT",
    "DEFAULT_HTTP_HOST",
    # Handler type constants
    "HANDLER_TYPE_HTTP",
    "HANDLER_TYPE_DATABASE",
    "HANDLER_TYPE_KAFKA",
    "HANDLER_TYPE_VAULT",
    "HANDLER_TYPE_CONSUL",
    "HANDLER_TYPE_VALKEY",
    "HANDLER_TYPE_GRPC",
    # Event bus kind constants
    "EVENT_BUS_INMEMORY",
    "EVENT_BUS_KAFKA",
    # Error class
    "RegistryError",
    # Registry classes
    "ProtocolBindingRegistry",
    "EventBusBindingRegistry",
    # Singleton accessors
    "get_handler_registry",
    "get_event_bus_registry",
    # Convenience functions
    "get_handler_class",
    "get_event_bus_class",
    "register_handlers_from_config",
    # Wiring functions
    "wire_default_handlers",
    "wire_handlers_from_contract",
    "get_known_handler_types",
    "get_known_event_bus_kinds",
    "wire_custom_handler",
    "wire_custom_event_bus",
    # Policy protocol and registry
    "ProtocolPolicy",
    "PolicyRegistry",
    # Envelope validation
    "PAYLOAD_REQUIRED_OPERATIONS",
    "validate_envelope",
    # Message dispatch engine
    "MessageDispatchEngine",
    "DispatcherRegistry",
    "ProtocolMessageDispatcher",
    # Context enforcement
    "DispatchContextEnforcer",
    # Message type registry (OMN-937)
    "MessageTypeRegistry",
    "MessageTypeRegistryError",
    "ModelMessageTypeEntry",
    "ModelDomainConstraint",
    "ProtocolMessageTypeRegistry",
    # Chain-aware dispatch (OMN-951)
    "ChainAwareDispatcher",
    "propagate_chain_context",
    "validate_dispatch_chain",
]
