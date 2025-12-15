# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime module for omnibase_infra.

This module provides the runtime infrastructure for the ONEX infrastructure layer,
including:

- Kernel: Contract-driven bootstrap entrypoint for the ONEX runtime
- ProtocolBindingRegistry: SINGLE SOURCE OF TRUTH for protocol handler registration
- EventBusBindingRegistry: Registry for event bus implementations
- RuntimeHostProcess: The infrastructure-specific runtime host process implementation
- Wiring functions: Register handlers and event buses with registries

The runtime module serves as the entry point for running infrastructure services
and configuring the handler ecosystem.
"""

from __future__ import annotations

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
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.wiring import (
    get_known_event_bus_kinds,
    get_known_handler_types,
    wire_custom_event_bus,
    wire_custom_handler,
    wire_default_handlers,
    wire_handlers_from_contract,
)

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
    # Envelope validation
    "PAYLOAD_REQUIRED_OPERATIONS",
    "validate_envelope",
]
