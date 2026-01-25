# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: PLW0603
# PLW0603 disabled: Global statement is intentional for singleton pattern with thread-safe initialization
"""Handler Registry - Constants and singleton accessors for handler registration.

This module provides constants and singleton accessor functions for the
RegistryProtocolBinding and RegistryEventBusBinding classes. The actual
registry implementations are in the runtime/registry/ directory.

Registry Classes (imported from runtime/registry/):
- RegistryProtocolBinding: Handler registration and resolution
- RegistryError: Error raised when registry operations fail
- RegistryEventBusBinding: Event bus implementation registration

Handler Type Constants:
- HANDLER_TYPE_HTTP, HANDLER_TYPE_DATABASE, etc.

Event Bus Kind Constants:
- EVENT_BUS_INMEMORY, EVENT_BUS_KAFKA

Singleton Accessors:
- get_handler_registry(): Returns singleton RegistryProtocolBinding
- get_event_bus_registry(): Returns singleton RegistryEventBusBinding

Example Usage:
    ```python
    from omnibase_infra.runtime.handler_registry import (
        RegistryProtocolBinding,
        HANDLER_TYPE_HTTP,
        HANDLER_TYPE_DATABASE,
    )

    registry = RegistryProtocolBinding()

    # Register handlers
    registry.register(HANDLER_TYPE_HTTP, HttpHandler)
    registry.register(HANDLER_TYPE_DATABASE, PostgresHandler)

    # Resolve handlers
    handler_cls = registry.get(HANDLER_TYPE_HTTP)
    handler = handler_cls()

    # Check registration
    if registry.is_registered(HANDLER_TYPE_KAFKA):
        kafka_handler = registry.get(HANDLER_TYPE_KAFKA)

    # List all registered protocols
    protocols = registry.list_protocols()
    ```

Integration Points:
- RuntimeHostProcess uses this registry to discover and instantiate handlers
- Handlers are loaded based on contract definitions
- Supports hot-reload patterns for development
- Event bus registry enables runtime bus selection
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from omnibase_infra.runtime.models import ModelProtocolRegistrationConfig

# Import registry classes from their canonical locations
from omnibase_infra.runtime.registry.registry_event_bus_binding import (
    RegistryEventBusBinding,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    RegistryError,
    RegistryProtocolBinding,
)

if TYPE_CHECKING:
    from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
    from omnibase_infra.protocols import ProtocolContainerAware

# =============================================================================
# Handler Type Constants
# =============================================================================
# These string literals serve as protocol type identifiers for handler registration.
# Will be replaced with EnumHandlerType after omnibase_core merge.

HANDLER_TYPE_HTTP: str = "http"
"""HTTP/REST API protocol handler type."""

HANDLER_TYPE_DATABASE: str = "db"
"""Database (PostgreSQL, etc.) protocol handler type.

Note: Value is "db" to match operation prefixes (db.query, db.execute).
Operations are routed by extracting the prefix before the first dot."""

HANDLER_TYPE_KAFKA: str = "kafka"
"""Kafka message broker protocol handler type."""

HANDLER_TYPE_VAULT: str = "vault"
"""HashiCorp Vault secret management protocol handler type."""

HANDLER_TYPE_CONSUL: str = "consul"
"""HashiCorp Consul service discovery protocol handler type."""

HANDLER_TYPE_VALKEY: str = "valkey"
"""Valkey (Redis-compatible) cache/message protocol handler type.

Note: Value is "valkey" to match operation prefixes (valkey.get, valkey.set).
Valkey is a Redis-compatible fork; we use valkey-py (redis-py compatible)."""

HANDLER_TYPE_GRPC: str = "grpc"
"""gRPC protocol handler type."""

HANDLER_TYPE_MCP: str = "mcp"
"""MCP (Model Context Protocol) handler type for AI agent tool integration.

The MCP handler exposes ONEX nodes as tools for AI agents via streamable HTTP.
Supports tools/list and tools/call operations per the MCP specification."""

HANDLER_TYPE_GRAPH: str = "graph"
"""Graph database (Memgraph/Neo4j) protocol handler type."""

HANDLER_TYPE_INTENT: str = "intent"  # DEMO (OMN-1515)
"""Intent storage and query handler type for demo wiring."""


# =============================================================================
# Event Bus Kind Constants
# =============================================================================

EVENT_BUS_INMEMORY: str = "inmemory"
"""In-memory event bus for local/testing deployments."""

EVENT_BUS_KAFKA: str = "kafka"
"""Kafka-based distributed event bus (Beta)."""


# =============================================================================
# Module-Level Singleton Registries
# =============================================================================

# Module-level singleton instances (lazy initialized)
_handler_registry: RegistryProtocolBinding | None = None
_event_bus_registry: RegistryEventBusBinding | None = None
_singleton_lock: threading.Lock = threading.Lock()


def get_handler_registry() -> RegistryProtocolBinding:
    """Get the singleton handler registry instance.

    Returns a module-level singleton instance of RegistryProtocolBinding.
    Creates the instance on first call (lazy initialization).

    Returns:
        RegistryProtocolBinding: The singleton handler registry instance.

    Example:
        >>> registry = get_handler_registry()
        >>> registry.register(HANDLER_TYPE_HTTP, HttpHandler)
        >>> same_registry = get_handler_registry()
        >>> same_registry is registry
        True
    """
    global _handler_registry
    if _handler_registry is None:
        with _singleton_lock:
            # Double-check locking pattern
            if _handler_registry is None:
                _handler_registry = RegistryProtocolBinding()
    return _handler_registry


def get_event_bus_registry() -> RegistryEventBusBinding:
    """Get the singleton event bus registry instance.

    Returns a module-level singleton instance of RegistryEventBusBinding.
    Creates the instance on first call (lazy initialization).

    Returns:
        RegistryEventBusBinding: The singleton event bus registry instance.

    Example:
        >>> registry = get_event_bus_registry()
        >>> registry.register(EVENT_BUS_INMEMORY, EventBusInmemory)
        >>> same_registry = get_event_bus_registry()
        >>> same_registry is registry
        True
    """
    global _event_bus_registry
    if _event_bus_registry is None:
        with _singleton_lock:
            # Double-check locking pattern
            if _event_bus_registry is None:
                _event_bus_registry = RegistryEventBusBinding()
    return _event_bus_registry


# =============================================================================
# Convenience Functions
# =============================================================================


def get_handler_class(handler_type: str) -> type[ProtocolContainerAware]:
    """Get handler class for the given type from the singleton registry.

    Convenience function that wraps get_handler_registry().get().

    Args:
        handler_type: Protocol type identifier (e.g., HANDLER_TYPE_HTTP).

    Returns:
        Handler class registered for the protocol type.

    Raises:
        RegistryError: If handler_type is not registered.

    Example:
        >>> from omnibase_infra.runtime.handler_registry import (
        ...     get_handler_class,
        ...     HANDLER_TYPE_HTTP,
        ... )
        >>> handler_cls = get_handler_class(HANDLER_TYPE_HTTP)
        >>> handler = handler_cls()
    """
    return get_handler_registry().get(handler_type)


def get_event_bus_class(bus_kind: str) -> type[ProtocolEventBus]:
    """Get event bus class for the given kind from the singleton registry.

    Convenience function that wraps get_event_bus_registry().get().

    Args:
        bus_kind: Bus kind identifier (e.g., EVENT_BUS_INMEMORY).

    Returns:
        Event bus class registered for the bus kind.

    Raises:
        RuntimeHostError: If bus_kind is not registered.

    Example:
        >>> from omnibase_infra.runtime.handler_registry import (
        ...     get_event_bus_class,
        ...     EVENT_BUS_INMEMORY,
        ... )
        >>> bus_cls = get_event_bus_class(EVENT_BUS_INMEMORY)
        >>> bus = bus_cls()
    """
    return get_event_bus_registry().get(bus_kind)


def register_handlers_from_config(
    runtime: object,  # Will be BaseRuntimeHostProcess
    protocol_configs: list[ModelProtocolRegistrationConfig],
) -> None:
    """Register protocol handlers from configuration.

    Called by BaseRuntimeHostProcess to wire up handlers based on contract config.
    This function validates and processes protocol registration configurations,
    registering the appropriate handlers with the runtime.

    Args:
        runtime: The runtime host process instance (BaseRuntimeHostProcess).
            Typed as object temporarily until BaseRuntimeHostProcess is implemented.
        protocol_configs: List of ModelProtocolRegistrationConfig instances from contract.
            Each config specifies type, protocol_class, enabled flag, and options.

    Example:
        >>> from omnibase_infra.runtime.models import ModelProtocolRegistrationConfig
        >>> protocol_configs = [
        ...     ModelProtocolRegistrationConfig(
        ...         type="http", protocol_class="HttpHandler", enabled=True
        ...     ),
        ...     ModelProtocolRegistrationConfig(
        ...         type="db", protocol_class="PostgresHandler", enabled=True
        ...     ),
        ... ]
        >>> register_handlers_from_config(runtime, protocol_configs)

    Note:
        **Placeholder implementation** - only validates config structure.

        TODO(OMN-41): Implement full handler resolution:
        1. Use importlib to resolve protocol_class string to actual class
        2. Validate class implements ProtocolContainerAware protocol
        3. Register handler with runtime via get_handler_registry()
        4. Support handler instantiation options from config.options
    """
    # Placeholder: validate config structure only, defer registration to OMN-41
    for config in protocol_configs:
        if not config.enabled:
            continue

        if config.type and config.protocol_class:
            # Config structure is valid - actual resolution deferred to OMN-41
            _ = config  # Explicit acknowledgment that config is intentionally unused


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    # Event bus kind constants
    "EVENT_BUS_INMEMORY",
    "EVENT_BUS_KAFKA",
    "HANDLER_TYPE_CONSUL",
    "HANDLER_TYPE_DATABASE",
    "HANDLER_TYPE_GRPC",
    "HANDLER_TYPE_GRAPH",
    # Handler type constants
    "HANDLER_TYPE_HTTP",
    "HANDLER_TYPE_INTENT",
    "HANDLER_TYPE_KAFKA",
    "HANDLER_TYPE_MCP",
    "HANDLER_TYPE_VALKEY",
    "HANDLER_TYPE_VAULT",
    # Error class
    "RegistryError",
    # Registry classes
    "RegistryEventBusBinding",
    "RegistryProtocolBinding",
    "get_event_bus_class",
    "get_event_bus_registry",
    # Convenience functions
    "get_handler_class",
    # Singleton accessors
    "get_handler_registry",
    "register_handlers_from_config",
]
