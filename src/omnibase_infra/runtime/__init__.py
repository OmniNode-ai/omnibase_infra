# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime module for omnibase_infra.

This module provides the runtime infrastructure for the ONEX infrastructure layer,
including three SINGLE SOURCE OF TRUTH registries and the runtime execution host.

Core Registries
---------------
- **PolicyRegistry**: SINGLE SOURCE OF TRUTH for policy plugin registration
    - Container-based DI support (preferred) or singleton accessor (legacy)
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
- **Wiring functions**: Register handlers and event buses with registries
- **Envelope validation**: Validate event envelope structures

Message Dispatch Engine
-----------------------
- **MessageDispatchEngine**: Runtime dispatch engine for message routing
- **RegistryDispatcher**: Thread-safe registry for dispatchers with freeze pattern
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
    ProtocolMessageDispatcher,
    RegistryDispatcher,
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

# NOTE: Registration dispatchers (DispatcherNodeIntrospected, DispatcherRuntimeTick,
# DispatcherNodeRegistrationAcked) and IntrospectionEventRouter moved to registration
# domain (OMN-1346). Import directly from:
#   omnibase_infra.nodes.node_registration_orchestrator.dispatchers
#   omnibase_infra.nodes.node_registration_orchestrator.introspection_event_router

# Handler plugin loader (OMN-1132)
from omnibase_infra.runtime.handler_plugin_loader import (
    CONTRACT_YAML_FILENAME,
    HANDLER_CONTRACT_FILENAME,
    HandlerPluginLoader,
    MAX_CONTRACT_SIZE,
)
from omnibase_infra.runtime.protocol_handler_plugin_loader import (
    ProtocolHandlerPluginLoader,
)

# Handler discovery protocol and implementation (OMN-1133)
from omnibase_infra.runtime.protocol_handler_discovery import (
    ProtocolHandlerDiscovery,
)
from omnibase_infra.runtime.contract_handler_discovery import (
    ContractHandlerDiscovery,
)

# Projector plugin loading and schema validation (OMN-1168, OMN-1169)
from omnibase_infra.runtime.projector_plugin_loader import (
    ProjectorPluginLoader,
)
from omnibase_infra.runtime.projector_schema_manager import (
    ProjectorSchemaError,
    ProjectorSchemaValidator,
)
from omnibase_infra.runtime.projector_shell import ProjectorShell

# Invocation security enforcer (OMN-1098)
from omnibase_infra.runtime.invocation_security_enforcer import (
    InvocationSecurityEnforcer,
    SecurityViolationError,
)

# Security metadata validator (OMN-1137)
from omnibase_infra.runtime.security_metadata_validator import (
    SecurityMetadataValidator,
    validate_handler_security,
)

# Chain-aware dispatch (OMN-951) - must be imported LAST to avoid circular import
from omnibase_infra.runtime.chain_aware_dispatch import (
    ChainAwareDispatcher,
    propagate_chain_context,
    validate_dispatch_chain,
)

# isort: on

__all__: list[str] = [
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
    # Dispatcher registry
    "RegistryDispatcher",
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
    # NOTE: DispatcherNodeIntrospected, DispatcherRuntimeTick, DispatcherNodeRegistrationAcked,
    # and IntrospectionEventRouter moved to registration domain (OMN-1346)
    # Handler plugin loader (OMN-1132)
    "CONTRACT_YAML_FILENAME",
    "HANDLER_CONTRACT_FILENAME",
    "HandlerPluginLoader",
    "MAX_CONTRACT_SIZE",
    "ProtocolHandlerPluginLoader",
    # Handler discovery protocol and implementation (OMN-1133)
    "ContractHandlerDiscovery",
    "ProtocolHandlerDiscovery",
    # Projector schema validation (OMN-1168) and shell (OMN-1169)
    "ProjectorSchemaError",
    "ProjectorPluginLoader",
    "ProjectorSchemaValidator",
    "ProjectorShell",
    # Invocation security enforcer (OMN-1098)
    "InvocationSecurityEnforcer",
    "SecurityViolationError",
    # Security metadata validator (OMN-1137)
    "SecurityMetadataValidator",
    "validate_handler_security",
]
