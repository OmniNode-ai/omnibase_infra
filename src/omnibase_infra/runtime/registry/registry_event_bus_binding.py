# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Binding Registry - Registry for event bus implementations.

This module provides the RegistryEventBusBinding class for registering and
resolving event bus implementations in the omnibase_infra layer.

Event Bus Categories:
- InMemory: Local in-process event bus for testing and simple deployments
- Kafka: Distributed event bus for production deployments (Beta)

Example Usage:
    ```python
    from omnibase_infra.runtime.registry import RegistryEventBusBinding
    from omnibase_infra.runtime.handler_registry import (
        EVENT_BUS_INMEMORY,
        EVENT_BUS_KAFKA,
    )

    registry = RegistryEventBusBinding()
    registry.register(EVENT_BUS_INMEMORY, EventBusInmemory)

    if registry.is_registered(EVENT_BUS_INMEMORY):
        bus_cls = registry.get(EVENT_BUS_INMEMORY)
        bus = bus_cls()
    ```

Integration Points:
- RuntimeHostProcess uses this registry to select event bus implementations
- Event bus selection is based on deployment configuration
- Supports runtime bus selection for different environments
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from omnibase_infra.errors import EventBusRegistryError, ModelInfraErrorContext

if TYPE_CHECKING:
    from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus


class RegistryEventBusBinding:
    """Registry for event bus implementations.

    Provides a centralized registry for event bus types, enabling runtime
    selection of event bus implementations based on deployment configuration.

    This registry is thread-safe and supports concurrent registration and
    retrieval operations.

    Note:
        Unlike RegistryProtocolBinding, this registry does not provide
        unregister() or clear() methods. Event buses are infrastructure
        components that should remain registered for the lifetime of the
        application. Removing them at runtime could cause message routing
        failures and system instability. Event bus registrations are
        permanent for the runtime lifecycle to ensure consistent message
        delivery throughout the application's execution.

    Attributes:
        _registry: Internal storage mapping bus_kind to bus class.
        _lock: Threading lock for thread-safe operations.

    Example:
        ```python
        registry = RegistryEventBusBinding()
        registry.register("inmemory", EventBusInmemory)

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

        Validation Order:
            1. Protocol method existence - verifies bus_cls has publish_envelope()
               or publish() method via hasattr() checks
            2. Method callability - verifies the publish method is actually callable,
               not a non-callable attribute
            3. Duplicate detection - prevents re-registration of existing bus_kind;
               performed under lock for thread safety
            4. Thread-safe registration - stores binding under lock

        Note:
            Unlike RegistryProtocolBinding, this registry prevents overwriting
            existing registrations. Event buses are infrastructure components
            that should remain stable for the application lifetime.

        Pydantic vs Registry Validation:
            This registry uses **runtime duck typing** for protocol validation,
            not Pydantic models. This approach allows any class implementing the
            required methods to be registered, regardless of inheritance hierarchy.

        Why Signature Validation is Not Used:
            We intentionally do NOT use ``inspect.signature()`` to validate
            method signatures. This design decision supports ONEX architecture:

            1. **Event bus signature variance is intentional**: Different event bus
               implementations may accept various envelope types. Enforcing a
               specific signature would break valid implementations.

            2. **Duck typing is ONEX principle**: Per CLAUDE.md, protocol resolution
               uses duck typing through protocols, never isinstance. Signature
               validation would conflict with this design.

            3. **Generic type flexibility**: ``ProtocolEventBusLike`` uses ``object``
               for generic payloads to allow flexibility across implementations.

            4. **Wrapper compatibility**: Decorated methods, wrappers with
               ``*args, **kwargs``, and generic signatures would fail signature
               validation despite being valid implementations.

            Method existence + callability is sufficient for duck typing validation.

        Args:
            bus_kind: Unique identifier for the bus type (e.g., "inmemory", "kafka").
            bus_cls: Event bus class implementing ProtocolEventBus protocol.

        Raises:
            EventBusRegistryError: If bus_cls does not implement required ProtocolEventBus
                methods (missing ``publish_envelope()`` or ``publish()``, or methods
                are not callable). Also raised if bus_kind is already registered.

        Example:
            ```python
            registry.register(EVENT_BUS_INMEMORY, EventBusInmemory)
            registry.register(EVENT_BUS_KAFKA, EventBusKafka)
            ```
        """
        # Validate bus_cls implements ProtocolEventBus
        has_publish_envelope = hasattr(bus_cls, "publish_envelope")
        has_publish = hasattr(bus_cls, "publish")

        if not has_publish_envelope and not has_publish:
            raise EventBusRegistryError(
                f"Event bus class {bus_cls.__name__} is missing "
                f"'publish_envelope()' or 'publish()' method from "
                f"ProtocolEventBus protocol",
                bus_kind=bus_kind,
                bus_class=bus_cls.__name__,
                context=ModelInfraErrorContext.with_correlation(
                    operation="register",
                ),
            )

        # Check that at least one publish method is callable
        if has_publish_envelope:
            if not callable(getattr(bus_cls, "publish_envelope", None)):
                raise EventBusRegistryError(
                    f"Event bus class {bus_cls.__name__} has "
                    f"'publish_envelope' attribute but it is not callable",
                    bus_kind=bus_kind,
                    bus_class=bus_cls.__name__,
                    context=ModelInfraErrorContext.with_correlation(
                        operation="register",
                    ),
                )

        if has_publish:
            if not callable(getattr(bus_cls, "publish", None)):
                raise EventBusRegistryError(
                    f"Event bus class {bus_cls.__name__} has "
                    f"'publish' attribute but it is not callable",
                    bus_kind=bus_kind,
                    bus_class=bus_cls.__name__,
                    context=ModelInfraErrorContext.with_correlation(
                        operation="register",
                    ),
                )

        with self._lock:
            if bus_kind in self._registry:
                raise EventBusRegistryError(
                    f"Event bus kind '{bus_kind}' is already registered",
                    bus_kind=bus_kind,
                    existing_class=self._registry[bus_kind].__name__,
                    context=ModelInfraErrorContext.with_correlation(
                        operation="register",
                    ),
                )
            self._registry[bus_kind] = bus_cls

    def get(self, bus_kind: str) -> type[ProtocolEventBus]:
        """Retrieve a registered event bus class.

        Args:
            bus_kind: Identifier of the bus type to retrieve.

        Returns:
            The event bus class registered for the given bus_kind.

        Raises:
            EventBusRegistryError: If bus_kind is not registered.

        Example:
            ```python
            bus_cls = registry.get(EVENT_BUS_INMEMORY)
            bus = bus_cls()
            ```
        """
        with self._lock:
            if bus_kind not in self._registry:
                available = list(self._registry.keys())
                raise EventBusRegistryError(
                    f"Event bus kind '{bus_kind}' is not registered",
                    bus_kind=bus_kind,
                    available_kinds=available,
                    context=ModelInfraErrorContext.with_correlation(
                        operation="get",
                    ),
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


__all__: list[str] = [
    "RegistryEventBusBinding",
]
