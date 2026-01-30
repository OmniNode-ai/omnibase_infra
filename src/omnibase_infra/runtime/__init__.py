# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime module for omnibase_infra.

This module provides the runtime infrastructure for the ONEX infrastructure layer,
including three SINGLE SOURCE OF TRUTH registries and the runtime execution host.

Core Registries
---------------
- **RegistryPolicy**: SINGLE SOURCE OF TRUTH for policy plugin registration
    - Container-based DI support (preferred) or singleton accessor (legacy)
    - Thread-safe registration by (policy_id, policy_type, version)
    - Enforces synchronous-by-default execution (async must be explicit)
    - Supports orchestrator and reducer policy types with version resolution
    - Pure decision logic plugins (no I/O, no side effects)
    - Integrates with ModelOnexContainer for DI pattern

- **RegistryProtocolBinding**: SINGLE SOURCE OF TRUTH for protocol handler registration
    - Maps handler types to handler implementations
    - Enables protocol-based dependency injection
    - Supports HTTP, database, Kafka, Vault, Consul, Valkey/Redis, gRPC handlers

- **RegistryEventBusBinding**: Registry for event bus implementations
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
    RegistryError,
    RegistryEventBusBinding,
    RegistryProtocolBinding,
    get_event_bus_class,
    get_event_bus_registry,
    get_handler_class,
    get_handler_registry,
    register_handlers_from_config,
)

from omnibase_infra.runtime.service_kernel import bootstrap as kernel_bootstrap
from omnibase_infra.runtime.service_kernel import load_runtime_config
from omnibase_infra.runtime.service_kernel import main as kernel_main
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.models import (
    ModelContractLoadResult,
    ModelProjectorNotificationConfig,
    ModelRuntimeContractConfig,
    ModelRuntimeSchedulerConfig,
    ModelRuntimeSchedulerMetrics,
    ModelRuntimeTick,
    ModelSecurityConfig,
    ModelStateTransitionNotification,
    ModelTransitionNotificationPublisherMetrics,
)
from omnibase_infra.runtime.registry_policy import RegistryPolicy
from omnibase_infra.runtime.protocol_policy import ProtocolPolicy
from omnibase_infra.runtime.protocols import (
    ProtocolRuntimeScheduler,
    ProtocolTransitionNotificationPublisher,
)
from omnibase_infra.runtime.mixins import (
    ProtocolProjectorNotificationContext,
)
from omnibase_infra.runtime.registry import (
    MessageTypeRegistryError,
    ModelDomainConstraint,
    ModelMessageTypeEntry,
    ProtocolMessageTypeRegistry,
    RegistryMessageType,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.runtime_scheduler import RuntimeScheduler
from omnibase_infra.runtime.util_wiring import (
    get_known_event_bus_kinds,
    get_known_handler_types,
    wire_custom_event_bus,
    wire_custom_handler,
    wire_default_handlers,
    wire_handlers_from_contract,
)

# Container wiring (OMN-888)
from omnibase_infra.runtime.util_container_wiring import (
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

# Handler bootstrap source (OMN-1087)
from omnibase_infra.runtime.handler_bootstrap_source import (
    HandlerBootstrapSource,
    SOURCE_TYPE_BOOTSTRAP,
)

# Handler identity helper (OMN-1095)
from omnibase_infra.runtime.handler_identity import (
    HANDLER_IDENTITY_PREFIX,
    handler_identity,
)

# Handler source resolver (OMN-1095)
from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver

# Handler contract config loader
from omnibase_infra.runtime.handler_contract_config_loader import (
    MAX_CONTRACT_SIZE_BYTES,
    extract_handler_config,
    load_handler_contract_config,
)

# Binding config resolver (OMN-765)
from omnibase_infra.runtime.binding_config_resolver import BindingConfigResolver
from omnibase_infra.runtime.protocol_handler_plugin_loader import (
    ProtocolHandlerPluginLoader,
)

# Binding expression resolver (OMN-1518)
from omnibase_infra.runtime.binding_resolver import (
    BindingExpressionParser,
    OperationBindingResolver,
    MAX_EXPRESSION_LENGTH,
    MAX_PATH_SEGMENTS,
    VALID_CONTEXT_PATHS,
    VALID_SOURCES,
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

# Transition notification publisher and outbox (OMN-1139)
from omnibase_infra.runtime.constants_notification import FROM_STATE_INITIAL
from omnibase_infra.runtime.transition_notification_publisher import (
    TransitionNotificationPublisher,
)
from omnibase_infra.runtime.transition_notification_outbox import (
    TransitionNotificationOutbox,
)

# Topic-scoped publisher (OMN-1621)
from omnibase_infra.runtime.publisher_topic_scoped import PublisherTopicScoped

# Event bus subcontract wiring (OMN-1621)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
    load_event_bus_subcontract,
)

# Runtime contract config loader (OMN-1519)
from omnibase_infra.runtime.runtime_contract_config_loader import (
    RuntimeContractConfigLoader,
)

# Security constants (OMN-1519)
from omnibase_infra.runtime.constants_security import (
    ALLOW_NAMESPACE_OVERRIDE_ENV_VAR,
    SECURITY_CONFIG_PATH_ENV_VAR,
    TRUSTED_HANDLER_NAMESPACE_PREFIXES,
)

# Registry contract source (OMN-1100)
from omnibase_infra.runtime.registry_contract_source import (
    DEFAULT_CONSUL_HOST,
    DEFAULT_CONSUL_PORT,
    DEFAULT_CONTRACT_PREFIX,
    RegistryContractSource,
    adelete_contract_from_consul,
    alist_contracts_in_consul,
    astore_contract_in_consul,
    delete_contract_from_consul,
    list_contracts_in_consul,
    store_contract_in_consul,
)

# Kafka contract source (OMN-1654)
from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

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
    # Message dispatch engine
    "MessageDispatchEngine",
    # Message type registry (OMN-937)
    "MessageTypeRegistryError",
    "ModelContractLoadResult",
    "ModelDomainConstraint",
    "ModelMessageTypeEntry",
    "ModelProjectorNotificationConfig",
    "ModelRuntimeContractConfig",
    "ModelRuntimeSchedulerConfig",
    "ModelRuntimeSchedulerMetrics",
    "ModelRuntimeTick",
    "ModelStateTransitionNotification",
    "ModelTransitionNotificationPublisherMetrics",
    "ProtocolMessageDispatcher",
    "ProtocolMessageTypeRegistry",
    # Registry classes
    "RegistryEventBusBinding",
    "RegistryMessageType",
    "RegistryPolicy",
    "RegistryProtocolBinding",
    # Policy protocol and registry
    "ProtocolPolicy",
    # Dispatcher registry
    "RegistryDispatcher",
    # Runtime scheduler (OMN-953)
    "ProtocolRuntimeScheduler",
    # Transition notification (OMN-1139)
    "ProtocolTransitionNotificationPublisher",
    # Projector notification context protocol (OMN-1139)
    "ProtocolProjectorNotificationContext",
    # Error class
    "RegistryError",
    # Notification constants (OMN-1139)
    "FROM_STATE_INITIAL",
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
    # Handler bootstrap source (OMN-1087)
    "HandlerBootstrapSource",
    "SOURCE_TYPE_BOOTSTRAP",
    # Handler identity helper (OMN-1095)
    "HANDLER_IDENTITY_PREFIX",
    "handler_identity",
    # Handler source resolver (OMN-1095)
    "HandlerSourceResolver",
    # Handler contract config loader
    "MAX_CONTRACT_SIZE_BYTES",
    "extract_handler_config",
    "load_handler_contract_config",
    # Binding config resolver (OMN-765)
    "BindingConfigResolver",
    # Binding expression resolver (OMN-1518)
    "BindingExpressionParser",
    "OperationBindingResolver",
    "MAX_EXPRESSION_LENGTH",
    "MAX_PATH_SEGMENTS",
    "VALID_CONTEXT_PATHS",
    "VALID_SOURCES",
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
    # Transition notification publisher and outbox (OMN-1139)
    "TransitionNotificationOutbox",
    "TransitionNotificationPublisher",
    # Topic-scoped publisher (OMN-1621)
    "PublisherTopicScoped",
    # Event bus subcontract wiring (OMN-1621)
    "EventBusSubcontractWiring",
    "load_event_bus_subcontract",
    # Runtime contract config loader (OMN-1519)
    "RuntimeContractConfigLoader",
    # Security constants and configuration (OMN-1519)
    "ALLOW_NAMESPACE_OVERRIDE_ENV_VAR",
    "ModelSecurityConfig",
    "SECURITY_CONFIG_PATH_ENV_VAR",
    "TRUSTED_HANDLER_NAMESPACE_PREFIXES",
    # Registry contract source (OMN-1100)
    "DEFAULT_CONSUL_HOST",
    "DEFAULT_CONSUL_PORT",
    "DEFAULT_CONTRACT_PREFIX",
    "RegistryContractSource",
    "adelete_contract_from_consul",
    "alist_contracts_in_consul",
    "astore_contract_in_consul",
    "delete_contract_from_consul",
    "list_contracts_in_consul",
    "store_contract_in_consul",
    # Kafka contract source (OMN-1654)
    "KafkaContractSource",
]
