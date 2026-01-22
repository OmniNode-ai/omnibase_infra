# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol Binding Registry - SINGLE SOURCE OF TRUTH for handler registration.

This module provides the RegistryProtocolBinding class which implements the
ProtocolContainerAwareRegistry protocol from omnibase_spi. It serves as the
centralized location for registering and resolving protocol handlers
in the omnibase_infra layer.

The registry is responsible for:
- Registering infrastructure handlers by protocol type identifier
- Resolving handler classes for protocol types (requires ProtocolContainerAware)
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
        RegistryProtocolBinding,
    )
    from omnibase_infra.runtime.handler_registry import (
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
"""

from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, Any, cast

from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolContainerAware


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
        >>> registry = RegistryProtocolBinding()
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

        # NOTE: Cast required for mypy - **dict[str, object] doesn't satisfy **context: Any
        super().__init__(
            message=message,
            context=context,
            **cast("dict[str, Any]", extra_context),
        )


# =============================================================================
# Handler Registry
# =============================================================================


class RegistryProtocolBinding:
    """SINGLE SOURCE OF TRUTH for handler registration in omnibase_infra.

    Thread-safe registry for protocol handlers. Implements ProtocolContainerAwareRegistry
    protocol from omnibase_spi.

    The registry maintains a mapping from protocol type identifiers (strings like
    "http", "db", "kafka") to handler classes that implement the ProtocolContainerAware
    protocol.

    TODO(OMN-40): Migrate handler signature from tuple[str, str] to structured model.
        Current implementation uses bare strings for protocol types. Should migrate
        to ModelHandlerKey(handler_type: str, handler_kind: str) for consistency
        with RegistryPolicy's ModelPolicyKey pattern and improved type safety.
        See: https://linear.app/omninode/issue/OMN-880

    Thread Safety:
        All registration operations are protected by a threading.Lock to ensure
        thread-safe access in concurrent environments.

    Attributes:
        _registry: Internal dictionary mapping protocol types to handler classes
        _lock: Threading lock for thread-safe registration operations

    Example:
        >>> registry = RegistryProtocolBinding()
        >>> registry.register("http", HttpHandler)
        >>> registry.register("db", PostgresHandler)
        >>> handler_cls = registry.get("http")
        >>> print(registry.list_protocols())
        ['db', 'http']
    """

    def __init__(self) -> None:
        """Initialize an empty handler registry with thread lock."""
        self._registry: dict[str, type[ProtocolContainerAware]] = {}
        self._lock: threading.Lock = threading.Lock()

    def register(
        self,
        protocol_type: str,
        handler_cls: type[ProtocolContainerAware],
    ) -> None:
        """Register a protocol handler.

        Associates a protocol type identifier with a handler class. If the protocol
        type is already registered, the existing registration is overwritten.

        Validation Order:
            1. Protocol method existence - verifies handler_cls has execute() or
               handle() method via hasattr() checks
            2. Method callability - verifies the handler method is actually callable,
               not a non-callable attribute
            3. Thread-safe registration - stores binding under lock; overwrites
               allowed for existing protocol types

        Note:
            Unlike RegistryEventBusBinding, this registry allows overwriting
            existing registrations. This enables hot-reload patterns during
            development and testing.

        Pydantic vs Registry Validation:
            This registry uses **runtime duck typing** for protocol validation,
            not Pydantic models. This approach allows any class implementing the
            required methods to be registered, regardless of inheritance hierarchy.

        Why Signature Validation is Not Used:
            We intentionally do NOT use ``inspect.signature()`` to validate
            method signatures. This design decision supports ONEX architecture:

            1. **Handler signature variance is intentional**: Different handlers
               accept different typed envelopes (e.g., ``ModelEventEnvelope[T]``
               with various payload types). Enforcing a specific signature would
               break valid implementations.

            2. **Duck typing is ONEX principle**: Per CLAUDE.md, protocol resolution
               uses duck typing through protocols, never isinstance. Signature
               validation would conflict with this design.

            3. **Generic type flexibility**: Protocols use ``object`` for generic
               payloads explicitly to allow flexibility (see ProtocolEventBusLike).

            4. **Wrapper compatibility**: Decorated methods, wrappers with
               ``*args, **kwargs``, and generic signatures would fail signature
               validation despite being valid implementations.

            Method existence + callability is sufficient for duck typing validation.

        Args:
            protocol_type: Protocol type identifier (e.g., 'http', 'db', 'kafka').
                          Should be one of the HANDLER_TYPE_* constants.
            handler_cls: Handler class implementing the ProtocolContainerAware protocol.

        Raises:
            RegistryError: If handler_cls does not implement the ProtocolContainerAware protocol
                          (missing or non-callable execute()/handle() method).

        Example:
            >>> registry = RegistryProtocolBinding()
            >>> registry.register(HANDLER_TYPE_HTTP, HttpHandler)
            >>> registry.register(HANDLER_TYPE_DATABASE, PostgresHandler)
        """
        # Runtime type validation: Ensure handler_cls implements ProtocolContainerAware protocol
        # Check if execute() or handle() method exists and is callable
        # Following RegistryEventBusBinding pattern of supporting alternative methods
        has_execute = hasattr(handler_cls, "execute")
        has_handle = hasattr(handler_cls, "handle")

        if not has_execute and not has_handle:
            raise RegistryError(
                f"Handler class {handler_cls.__name__!r} for protocol type "
                f"{protocol_type!r} is missing 'execute()' or 'handle()' method "
                f"from ProtocolContainerAware protocol",
                protocol_type=protocol_type,
                context=ModelInfraErrorContext.with_correlation(
                    operation="register",
                ),
                handler_class=handler_cls.__name__,
            )

        # Check that at least one handler method is callable
        if has_execute:
            if not callable(getattr(handler_cls, "execute", None)):
                raise RegistryError(
                    f"Handler class {handler_cls.__name__!r} for protocol type "
                    f"{protocol_type!r} has 'execute' attribute but it is not callable",
                    protocol_type=protocol_type,
                    context=ModelInfraErrorContext.with_correlation(
                        operation="register",
                    ),
                    handler_class=handler_cls.__name__,
                )

        if has_handle:
            if not callable(getattr(handler_cls, "handle", None)):
                raise RegistryError(
                    f"Handler class {handler_cls.__name__!r} for protocol type "
                    f"{protocol_type!r} has 'handle' attribute but it is not callable",
                    protocol_type=protocol_type,
                    context=ModelInfraErrorContext.with_correlation(
                        operation="register",
                    ),
                    handler_class=handler_cls.__name__,
                )

        with self._lock:
            self._registry[protocol_type] = handler_cls

    def get(
        self,
        protocol_type: str,
    ) -> type[ProtocolContainerAware]:
        """Get handler class for protocol type.

        Resolves the handler class registered for the given protocol type.

        Args:
            protocol_type: Protocol type identifier.

        Returns:
            Handler class registered for the protocol type.

        Raises:
            RegistryError: If protocol type is not registered.

        Example:
            >>> registry = RegistryProtocolBinding()
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
                    context=ModelInfraErrorContext.with_correlation(
                        operation="get",
                    ),
                    registered_protocols=registered,
                )
            return handler_cls

    def list_protocols(self) -> list[str]:
        """List registered protocol types.

        Returns:
            List of registered protocol type identifiers, sorted alphabetically.

        Example:
            >>> registry = RegistryProtocolBinding()
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
            >>> registry = RegistryProtocolBinding()
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
            >>> registry = RegistryProtocolBinding()
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

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Example:
            >>> registry = RegistryProtocolBinding()
            >>> registry.register("http", HttpHandler)
            >>> registry.clear()
            >>> registry.list_protocols()
            []
        """
        warnings.warn(
            "RegistryProtocolBinding.clear() is intended for testing only."
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with self._lock:
            self._registry.clear()

    def __len__(self) -> int:
        """Return the number of registered handlers.

        Returns:
            Number of registered protocol handlers.

        Example:
            >>> registry = RegistryProtocolBinding()
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
            >>> registry = RegistryProtocolBinding()
            >>> registry.register("http", HttpHandler)
            >>> "http" in registry
            True
            >>> "unknown" in registry
            False
        """
        return self.is_registered(protocol_type)


__all__: list[str] = [
    "RegistryProtocolBinding",
    "RegistryError",
]
