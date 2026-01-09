# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol Binding Registry - SINGLE SOURCE OF TRUTH for handler registration.

This module provides the ProtocolBindingRegistry class which implements the
ProtocolHandlerRegistry protocol from omnibase_spi. It serves as the
centralized location for registering and resolving protocol handlers
in the omnibase_infra layer.

The registry is responsible for:
- Registering protocol handlers by protocol type identifier
- Resolving handler classes for protocol types
- Thread-safe registration operations
- Listing all registered protocol types

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

Example Usage:
    ```python
    from omnibase_infra.runtime.registry import (
        ProtocolBindingRegistry,
    )
    from omnibase_infra.runtime.handler_registry import (
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
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError

if TYPE_CHECKING:
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler


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

        Raises:
            RegistryError: If handler_cls does not implement the ProtocolHandler protocol
                          (missing or non-callable execute() method).

        Example:
            >>> registry = ProtocolBindingRegistry()
            >>> registry.register(HANDLER_TYPE_HTTP, HttpHandler)
            >>> registry.register(HANDLER_TYPE_DATABASE, PostgresHandler)
        """
        # Runtime type validation: Ensure handler_cls implements ProtocolHandler protocol
        # Check if execute() method exists and is callable
        execute_attr = getattr(handler_cls, "execute", None)

        if execute_attr is None:
            raise RegistryError(
                f"Handler class {handler_cls.__name__!r} does not implement "
                f"ProtocolHandler protocol: missing 'execute()' method",
                protocol_type=protocol_type,
                handler_class=handler_cls.__name__,
            )

        if not callable(execute_attr):
            raise RegistryError(
                f"Handler class {handler_cls.__name__!r} does not implement "
                f"ProtocolHandler protocol: execute() method (not callable)",
                protocol_type=protocol_type,
                handler_class=handler_cls.__name__,
            )

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
                registered = sorted(self._registry.keys())
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


__all__: list[str] = [
    "ProtocolBindingRegistry",
    "RegistryError",
]
