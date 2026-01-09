# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Registry - Constants and singleton accessors for handler registration.

This module provides constants and singleton accessor functions for the
ProtocolBindingRegistry and EventBusBindingRegistry classes. The actual
registry implementations are in the runtime/registry/ directory.

Registry Classes (imported from runtime/registry/):
- ProtocolBindingRegistry: Handler registration and resolution
- RegistryError: Error raised when registry operations fail
- EventBusBindingRegistry: Event bus implementation registration

Handler Type Constants:
- HANDLER_TYPE_HTTP, HANDLER_TYPE_DATABASE, etc.

Event Bus Kind Constants:
- EVENT_BUS_INMEMORY, EVENT_BUS_KAFKA

Singleton Accessors:
- get_handler_registry(): Returns singleton ProtocolBindingRegistry
- get_event_bus_registry(): Returns singleton EventBusBindingRegistry

Example Usage:
    ```python
    from omnibase_infra.runtime.handler_registry import (
        ProtocolBindingRegistry,
        HANDLER_TYPE_HTTP,
        HANDLER_TYPE_DATABASE,
    )

    registry = ProtocolBindingRegistry()

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
    EventBusBindingRegistry,
)
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    ProtocolBindingRegistry,
    RegistryError,
)

if TYPE_CHECKING:
    from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

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
_handler_registry: ProtocolBindingRegistry | None = None
_event_bus_registry: EventBusBindingRegistry | None = None
_singleton_lock: threading.Lock = threading.Lock()


def get_handler_registry() -> ProtocolBindingRegistry:
    """Get the singleton handler registry instance.

    Returns a module-level singleton instance of ProtocolBindingRegistry.
    Creates the instance on first call (lazy initialization).

    Returns:
        ProtocolBindingRegistry: The singleton handler registry instance.

    Example:
        >>> registry = get_handler_registry()
        >>> registry.register(HANDLER_TYPE_HTTP, HttpHandler)
        >>> same_registry = get_handler_registry()
        >>> same_registry is registry
        True
    """
    global _handler_registry  # noqa: PLW0603
    if _handler_registry is None:
        with _singleton_lock:
            # Double-check locking pattern
            if _handler_registry is None:
                _handler_registry = ProtocolBindingRegistry()
    return _handler_registry


def get_event_bus_registry() -> EventBusBindingRegistry:
    """Get the singleton event bus registry instance.

    Returns a module-level singleton instance of EventBusBindingRegistry.
    Creates the instance on first call (lazy initialization).

    Returns:
        EventBusBindingRegistry: The singleton event bus registry instance.

    Example:
        >>> registry = get_event_bus_registry()
        >>> registry.register(EVENT_BUS_INMEMORY, InMemoryEventBus)
        >>> same_registry = get_event_bus_registry()
        >>> same_registry is registry
        True
    """
    global _event_bus_registry  # noqa: PLW0603
    if _event_bus_registry is None:
        with _singleton_lock:
            # Double-check locking pattern
            if _event_bus_registry is None:
                _event_bus_registry = EventBusBindingRegistry()
    return _event_bus_registry


# =============================================================================
# Convenience Functions
# =============================================================================


def get_handler_class(handler_type: str) -> type[ProtocolHandler]:
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
        This is a placeholder implementation. Full protocol class resolution
        will be implemented when BaseRuntimeHostProcess is available.
    """
    # NOTE: Registry access (get_handler_registry()) will be needed when
    # TODO(OMN-41) is implemented to resolve and register handler classes.
    for config in protocol_configs:
        if not config.enabled:
            continue

        if config.type and config.protocol_class:
            # TODO(OMN-41): Resolve handler class from name using importlib
            # For now, just validate config structure is correct
            # The actual handler instantiation will be done by RuntimeHostProcess
            pass  # Validation passed; handler resolution deferred to OMN-41


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
    # Handler type constants
    "HANDLER_TYPE_HTTP",
    "HANDLER_TYPE_KAFKA",
    "HANDLER_TYPE_VALKEY",
    "HANDLER_TYPE_VAULT",
    "EventBusBindingRegistry",
    # Registry classes
    "ProtocolBindingRegistry",
    # Error class
    "RegistryError",
    "get_event_bus_class",
    "get_event_bus_registry",
    # Convenience functions
    "get_handler_class",
    # Singleton accessors
    "get_handler_registry",
    "register_handlers_from_config",
]
