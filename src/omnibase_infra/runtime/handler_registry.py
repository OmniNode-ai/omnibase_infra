# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Registry - SINGLE SOURCE OF TRUTH for handler registration.

This module provides the ProtocolBindingRegistry class which implements the
ProtocolHandlerRegistry protocol from omnibase_spi. It serves as the
centralized location for registering and resolving protocol handlers
in the omnibase_infra layer.

The registry is responsible for:
- Registering protocol handlers by protocol type identifier
- Resolving handler classes for protocol types
- Thread-safe registration operations
- Listing all registered protocol types
- Managing event bus implementations and their registration

Design Principles:
- Single source of truth: All handler registrations go through this registry
- Explicit over implicit: No auto-discovery magic, handlers explicitly registered
- Type-safe: Full typing for handler registrations (no Any types)
- Thread-safe: Registration operations protected by lock
- Testable: Easy to mock and test handler configurations

Handler Categories (by protocol type):
- HTTP handlers: REST API integrations
- Database handlers: PostgreSQL, Valkey connections
- Message broker handlers: Kafka message processing
- Service discovery handlers: Consul integration
- Secret management handlers: Vault integration

Event Bus Categories:
- InMemory: Local in-process event bus for testing and simple deployments
- Kafka: Distributed event bus for production deployments (Beta)

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

from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.runtime.models import ModelProtocolRegistrationConfig

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
# Registry Error
# =============================================================================


class RegistryError(RuntimeHostError):
    """Error raised when handler registry operations fail.

    Used for:
    - Attempting to get an unregistered handler
    - Registration failures (if duplicate registration is disallowed)
    - Invalid protocol type identifiers

    Extends RuntimeHostError as this is an infrastructure-layer runtime concern.

    Example:
        >>> registry = ProtocolBindingRegistry()
        >>> try:
        ...     handler = registry.get("unknown_protocol")
        ... except RegistryError as e:
        ...     print(f"Handler not found: {e}")
    """

    def __init__(
        self,
        message: str,
        protocol_type: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize RegistryError.

        Args:
            message: Human-readable error message
            protocol_type: The protocol type that caused the error (if applicable)
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information
        """
        # Add protocol_type to extra_context if provided
        if protocol_type is not None:
            extra_context["protocol_type"] = protocol_type

        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


# =============================================================================
# Handler Registry
# =============================================================================


class ProtocolBindingRegistry:
    """SINGLE SOURCE OF TRUTH for handler registration in omnibase_infra.

    Thread-safe registry for protocol handlers. Implements ProtocolHandlerRegistry
    protocol from omnibase_spi.

    The registry maintains a mapping from protocol type identifiers (strings like
    "http", "db", "kafka") to handler classes that implement the ProtocolHandler
    protocol.

    TODO(OMN-40): Migrate handler signature from tuple[str, str] to structured model.
        Current implementation uses bare strings for protocol types. Should migrate
        to ModelHandlerKey(handler_type: str, handler_kind: str) for consistency
        with PolicyRegistry's ModelPolicyKey pattern and improved type safety.
        See: https://linear.app/omninode/issue/OMN-880

    Thread Safety:
        All registration operations are protected by a threading.Lock to ensure
        thread-safe access in concurrent environments.

    Attributes:
        _registry: Internal dictionary mapping protocol types to handler classes
        _lock: Threading lock for thread-safe registration operations

    Example:
        >>> registry = ProtocolBindingRegistry()
        >>> registry.register("http", HttpHandler)
        >>> registry.register("db", PostgresHandler)
        >>> handler_cls = registry.get("http")
        >>> print(registry.list_protocols())
        ['db', 'http']
    """

    def __init__(self) -> None:
        """Initialize an empty handler registry with thread lock."""
        self._registry: dict[str, type[ProtocolHandler]] = {}
        self._lock: threading.Lock = threading.Lock()

    def register(
        self,
        protocol_type: str,
        handler_cls: type[ProtocolHandler],
    ) -> None:
        """Register a protocol handler.

        Associates a protocol type identifier with a handler class. If the protocol
        type is already registered, the existing registration is overwritten.

        Args:
            protocol_type: Protocol type identifier (e.g., 'http', 'db', 'kafka').
                          Should be one of the HANDLER_TYPE_* constants.
            handler_cls: Handler class implementing the ProtocolHandler protocol.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register(HANDLER_TYPE_HTTP, HttpHandler)
            >>> registry.register(HANDLER_TYPE_DATABASE, PostgresHandler)
        """
        with self._lock:
            self._registry[protocol_type] = handler_cls

    def get(
        self,
        protocol_type: str,
    ) -> type[ProtocolHandler]:
        """Get handler class for protocol type.

        Resolves the handler class registered for the given protocol type.

        Args:
            protocol_type: Protocol type identifier.

        Returns:
            Handler class registered for the protocol type.

        Raises:
            RegistryError: If protocol type is not registered.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register("http", HttpHandler)
            >>> handler_cls = registry.get("http")
            >>> handler = handler_cls()
        """
        with self._lock:
            handler_cls = self._registry.get(protocol_type)

        if handler_cls is None:
            registered = self.list_protocols()
            raise RegistryError(
                f"No handler registered for protocol type: {protocol_type!r}. "
                f"Registered protocols: {registered}",
                protocol_type=protocol_type,
                registered_protocols=registered,
            )

        return handler_cls

    def list_protocols(self) -> list[str]:
        """List registered protocol types.

        Returns:
            List of registered protocol type identifiers, sorted alphabetically.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register("http", HttpHandler)
            >>> registry.register("db", PostgresHandler)
            >>> print(registry.list_protocols())
            ['db', 'http']
        """
        with self._lock:
            return sorted(self._registry.keys())

    def is_registered(self, protocol_type: str) -> bool:
        """Check if protocol type is registered.

        Args:
            protocol_type: Protocol type identifier.

        Returns:
            True if protocol type is registered, False otherwise.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register("http", HttpHandler)
            >>> registry.is_registered("http")
            True
            >>> registry.is_registered("unknown")
            False
        """
        with self._lock:
            return protocol_type in self._registry

    def unregister(self, protocol_type: str) -> bool:
        """Unregister a protocol handler.

        Removes the handler registration for the given protocol type.
        This is useful for testing and hot-reload scenarios.

        Args:
            protocol_type: Protocol type identifier to unregister.

        Returns:
            True if the protocol was unregistered, False if it wasn't registered.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register("http", HttpHandler)
            >>> registry.unregister("http")
            True
            >>> registry.unregister("http")
            False
        """
        with self._lock:
            if protocol_type in self._registry:
                del self._registry[protocol_type]
                return True
            return False

    def clear(self) -> None:
        """Clear all handler registrations.

        Removes all registered handlers from the registry.
        This is useful for testing scenarios.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register("http", HttpHandler)
            >>> registry.clear()
            >>> registry.list_protocols()
            []
        """
        with self._lock:
            self._registry.clear()

    def __len__(self) -> int:
        """Return the number of registered handlers.

        Returns:
            Number of registered protocol handlers.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> len(registry)
            0
            >>> registry.register("http", HttpHandler)
            >>> len(registry)
            1
        """
        with self._lock:
            return len(self._registry)

    def __contains__(self, protocol_type: str) -> bool:
        """Check if protocol type is registered using 'in' operator.

        Args:
            protocol_type: Protocol type identifier.

        Returns:
            True if protocol type is registered, False otherwise.

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register("http", HttpHandler)
            >>> "http" in registry
            True
            >>> "unknown" in registry
            False
        """
        return self.is_registered(protocol_type)


# =============================================================================
# Event Bus Registry
# =============================================================================


class EventBusBindingRegistry:
    """Registry for event bus implementations.

    Provides a centralized registry for event bus types, enabling runtime
    selection of event bus implementations based on deployment configuration.

    This registry is thread-safe and supports concurrent registration and
    retrieval operations.

    Attributes:
        _registry: Internal storage mapping bus_kind to bus class.
        _lock: Threading lock for thread-safe operations.

    Example:
        ```python
        registry = EventBusBindingRegistry()
        registry.register("inmemory", InMemoryEventBus)

        if registry.is_registered("inmemory"):
            bus_cls = registry.get("inmemory")
            bus = bus_cls()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty event bus registry."""
        self._registry: dict[str, type[ProtocolEventBus]] = {}
        self._lock: threading.Lock = threading.Lock()

    def register(
        self,
        bus_kind: str,
        bus_cls: type[ProtocolEventBus],
    ) -> None:
        """Register an event bus implementation.

        Associates a bus_kind identifier with an event bus class that
        implements ProtocolEventBus.

        Args:
            bus_kind: Unique identifier for the bus type (e.g., "inmemory", "kafka").
            bus_cls: Event bus class implementing ProtocolEventBus protocol.

        Raises:
            RuntimeHostError: If bus_kind is already registered.

        Example:
            ```python
            registry.register(EVENT_BUS_INMEMORY, InMemoryEventBus)
            registry.register(EVENT_BUS_KAFKA, KafkaEventBus)
            ```
        """
        with self._lock:
            if bus_kind in self._registry:
                raise RuntimeHostError(
                    f"Event bus kind '{bus_kind}' is already registered",
                    bus_kind=bus_kind,
                    existing_class=self._registry[bus_kind].__name__,
                )
            self._registry[bus_kind] = bus_cls

    def get(self, bus_kind: str) -> type[ProtocolEventBus]:
        """Retrieve a registered event bus class.

        Args:
            bus_kind: Identifier of the bus type to retrieve.

        Returns:
            The event bus class registered for the given bus_kind.

        Raises:
            RuntimeHostError: If bus_kind is not registered.

        Example:
            ```python
            bus_cls = registry.get(EVENT_BUS_INMEMORY)
            bus = bus_cls()
            ```
        """
        with self._lock:
            if bus_kind not in self._registry:
                available = list(self._registry.keys())
                raise RuntimeHostError(
                    f"Event bus kind '{bus_kind}' is not registered",
                    bus_kind=bus_kind,
                    available_kinds=available,
                )
            return self._registry[bus_kind]

    def list_bus_kinds(self) -> list[str]:
        """List all registered event bus kinds.

        Returns:
            List of registered bus_kind identifiers.

        Example:
            ```python
            kinds = registry.list_bus_kinds()
            # ['inmemory', 'kafka']
            ```
        """
        with self._lock:
            return list(self._registry.keys())

    def is_registered(self, bus_kind: str) -> bool:
        """Check if an event bus kind is registered.

        Args:
            bus_kind: Identifier to check for registration.

        Returns:
            True if the bus_kind is registered, False otherwise.

        Example:
            ```python
            if registry.is_registered(EVENT_BUS_KAFKA):
                bus_cls = registry.get(EVENT_BUS_KAFKA)
            ```
        """
        with self._lock:
            return bus_kind in self._registry


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
    registry = get_handler_registry()
    for config in protocol_configs:
        if not config.enabled:
            continue

        if config.type and config.protocol_class:
            # TODO(OMN-41): Resolve handler class from name using importlib
            # For now, just validate config structure is correct
            # The actual handler instantiation will be done by RuntimeHostProcess
            pass


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
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
]
