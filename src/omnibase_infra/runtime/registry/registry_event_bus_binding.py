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

from omnibase_infra.errors import RuntimeHostError

if TYPE_CHECKING:
    from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus


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


__all__: list[str] = [
    "EventBusBindingRegistry",
]
