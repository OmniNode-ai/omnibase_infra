# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Bus Binding Registry - Registry for event bus implementations.

This module provides the EventBusBindingRegistry class for registering and
resolving event bus implementations in the omnibase_infra layer.

Event Bus Categories:
- InMemory: Local in-process event bus for testing and simple deployments
- Kafka: Distributed event bus for production deployments (Beta)

Example Usage:
    ```python
    from omnibase_infra.runtime.registry import EventBusBindingRegistry
    from omnibase_infra.runtime.handler_registry import (
        EVENT_BUS_INMEMORY,
        EVENT_BUS_KAFKA,
    )

    registry = EventBusBindingRegistry()
    registry.register(EVENT_BUS_INMEMORY, InMemoryEventBus)

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


class EventBusBindingRegistry:
    """Registry for event bus implementations.

    Provides a centralized registry for event bus types, enabling runtime
    selection of event bus implementations based on deployment configuration.

    This registry is thread-safe and supports concurrent registration and
    retrieval operations.

    Note:
        Unlike ProtocolBindingRegistry, this registry does not provide
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

        Validation Order:
            Validations are performed in fail-fast order with cheap checks first:

            1. **Protocol method existence** (O(1) hasattr checks):
               Verifies bus_cls has either ``publish_envelope()`` or ``publish()``
               method. This is the cheapest check - just attribute lookup.

            2. **Method callability** (O(1) callable checks):
               If a publish method exists, verifies it is actually callable
               (not a non-callable attribute). Slightly more expensive than
               existence check but still O(1).

            3. **Duplicate registration** (O(1) dict lookup, under lock):
               Checks if bus_kind is already registered. This validation is
               performed last and under lock because:

               - It requires lock acquisition (more expensive than attribute checks)
               - We want to validate the class is well-formed before checking
                 if it can be registered, so error messages are accurate
               - Duplicate registration errors are more common during development
                 but protocol errors indicate deeper issues

        Pydantic vs Registry Validation:
            This registry uses **runtime duck typing** for protocol validation,
            not Pydantic models. The validation checks:

            - Method existence via ``hasattr()``
            - Method callability via ``callable()``

            This approach allows any class implementing the required methods to
            be registered, regardless of inheritance hierarchy. Pydantic is not
            involved in the registration validation process.

        Thread Safety:
            The duplicate registration check is performed under lock to ensure
            thread-safe concurrent registration. Protocol validation is performed
            outside the lock since it only inspects the class (immutable) and
            does not access shared state.

        Args:
            bus_kind: Unique identifier for the bus type (e.g., "inmemory", "kafka").
            bus_cls: Event bus class implementing ProtocolEventBus protocol.

        Raises:
            EventBusRegistryError: If bus_cls does not implement required ProtocolEventBus
                methods (missing ``publish_envelope()`` or ``publish()``, or methods
                are not callable). Also raised if bus_kind is already registered.

        Example:
            ```python
            registry.register(EVENT_BUS_INMEMORY, InMemoryEventBus)
            registry.register(EVENT_BUS_KAFKA, KafkaEventBus)
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
    "EventBusBindingRegistry",
]
