# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Wiring Module - Registers concrete handlers with the handler registry.

This module provides functions to wire up concrete handler implementations
with the ProtocolBindingRegistry and EventBusBindingRegistry. It serves as
the bridge between handler implementations and the registry system.

The wiring module is responsible for:
- Registering default handlers for standard protocol types
- Registering handlers based on contract configuration
- Validating that requested handler types are known and supported
- Providing a summary of registered handlers for debugging

Design Principles:
- Explicit wiring: All handler registrations are explicit, not auto-discovered
- Contract-driven: Supports wiring from contract configuration dicts
- Validation: Unknown handler types raise clear errors
- Idempotent: Re-wiring the same handler is safe (overwrites previous)

Adding New Handlers:
    To add a new handler to the system, follow these steps:

    1. Create a handler class implementing the ProtocolHandler protocol:

        ```python
        from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

        class MyCustomHandler:
            '''Handler for custom protocol operations.'''

            async def initialize(self, config: dict[str, object]) -> None:
                '''Initialize handler with configuration.'''
                self._config = config

            async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
                '''Execute operation from envelope and return response.'''
                # Handle the envelope and return response dict
                return {"success": True, "data": ...}
        ```

    2. Add the handler to _KNOWN_HANDLERS dict with a type constant:

        In handler_registry.py, add a constant:
            HANDLER_TYPE_CUSTOM = "custom"

        In this module, add to _KNOWN_HANDLERS:
            HANDLER_TYPE_CUSTOM: (MyCustomHandler, "Custom protocol handler"),

    3. For runtime registration without modifying _KNOWN_HANDLERS, use
       wire_custom_handler():

        ```python
        from omnibase_infra.runtime.wiring import wire_custom_handler

        wire_custom_handler("custom", MyCustomHandler)
        ```

Integration with RuntimeHostProcess:
    The wiring module and RuntimeHostProcess work together in a two-phase
    registration and instantiation flow:

    Phase 1 - Class Registration (wiring module):
        When wire_default_handlers() is called, handler CLASSES are registered
        with the singleton ProtocolBindingRegistry. At this point, no handler
        instances exist - only the class types are stored in the registry.

    Phase 2 - Instance Creation (RuntimeHostProcess):
        After wiring, RuntimeHostProcess._populate_handlers_from_registry():
        1. Gets each handler CLASS from the singleton registry
        2. Instantiates the handler class: handler_instance = handler_cls()
        3. Calls initialize(config) on the instance with runtime configuration
        4. Stores the instance in self._handlers for envelope routing

    This two-phase approach allows:
        - Centralized handler class registration via wiring
        - Lazy instantiation controlled by RuntimeHostProcess
        - Per-process configuration passed to handler.initialize()
        - Test injection of mock handlers before start() is called

Example Usage:
    ```python
    from omnibase_infra.runtime.wiring import (
        wire_default_handlers,
        wire_handlers_from_contract,
    )

    # Wire all default handlers
    summary = wire_default_handlers()
    print(f"Registered handlers: {summary['handlers']}")
    print(f"Registered event buses: {summary['event_buses']}")

    # Wire handlers from contract config
    contract_config = {
        "handlers": [
            {"type": "http", "enabled": True},
            {"type": "db", "enabled": True},
        ],
        "event_bus": {"kind": "inmemory"},
    }
    wire_handlers_from_contract(contract_config)
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.handlers.handler_consul import ConsulHandler
from omnibase_infra.handlers.handler_db import DbAdapter
from omnibase_infra.handlers.handler_http import HttpRestAdapter
from omnibase_infra.handlers.handler_vault import VaultAdapter
from omnibase_infra.runtime.handler_registry import (
    EVENT_BUS_INMEMORY,
    HANDLER_TYPE_CONSUL,
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_HTTP,
    HANDLER_TYPE_VAULT,
    EventBusBindingRegistry,
    ProtocolBindingRegistry,
    get_event_bus_registry,
    get_handler_registry,
)

if TYPE_CHECKING:
    from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

logger = logging.getLogger(__name__)

# Known handler types that can be wired.
#
# Pattern: handler_type_constant -> (handler_class, description)
#
# - handler_type_constant: String constant defined in handler_registry.py
#   (e.g., HANDLER_TYPE_HTTP = "http"). This is the key used to look up
#   and route envelopes to the correct handler.
#
# - handler_class: The class implementing ProtocolHandler protocol.
#   Must have async initialize(config) and async execute(envelope) methods.
#   The wiring module registers the CLASS; RuntimeHostProcess instantiates it.
#
# - description: Human-readable description for logging and debugging.
#   Appears in log messages when handlers are registered.
#
# To add a new handler:
# 1. Define HANDLER_TYPE_XXX constant in handler_registry.py
# 2. Import the handler class at the top of this module
# 3. Add entry below: HANDLER_TYPE_XXX: (XxxHandler, "Description"),
#
# NOTE: HttpRestAdapter and DbAdapter use legacy execute(envelope: dict) signature.
# They will be migrated to ProtocolHandler.execute(request, operation_config) in future.
# Type ignore comments suppress MyPy errors during MVP phase.
_KNOWN_HANDLERS: dict[str, tuple[type[ProtocolHandler], str]] = {
    HANDLER_TYPE_CONSUL: (ConsulHandler, "HashiCorp Consul service discovery handler"),  # type: ignore[dict-item]
    HANDLER_TYPE_DATABASE: (DbAdapter, "PostgreSQL database adapter"),  # type: ignore[dict-item]
    HANDLER_TYPE_HTTP: (HttpRestAdapter, "HTTP REST protocol adapter"),  # type: ignore[dict-item]
    HANDLER_TYPE_VAULT: (VaultAdapter, "HashiCorp Vault secret management handler"),  # type: ignore[dict-item]
}

# Known event bus kinds that can be wired
# Maps bus kind constant to (bus_class, description)
_KNOWN_EVENT_BUSES: dict[str, tuple[type[ProtocolEventBus], str]] = {
    EVENT_BUS_INMEMORY: (InMemoryEventBus, "In-memory event bus for local/testing"),
}


def wire_default_handlers() -> dict[str, list[str]]:
    """Register all default handlers and event buses with singleton registries.

    This function registers the standard set of handlers and event buses
    that are available for use by the RuntimeHostProcess. It is the primary
    way to initialize the handler ecosystem.

    Registered Handlers:
        - CONSUL: ConsulHandler for HashiCorp Consul service discovery
        - DB: DbAdapter for PostgreSQL database operations
        - HTTP: HttpRestAdapter for HTTP/REST protocol operations
        - VAULT: VaultAdapter for HashiCorp Vault secret management

    Registered Event Buses:
        - INMEMORY: InMemoryEventBus for local/testing deployments

    Returns:
        Summary dict with keys:
            - handlers: List of registered handler type names
            - event_buses: List of registered event bus kind names

    Example:
        >>> summary = wire_default_handlers()
        >>> print(summary)
        {'handlers': ['consul', 'db', 'http', 'vault'], 'event_buses': ['inmemory']}

    Note:
        This function uses the singleton registries returned by
        get_handler_registry() and get_event_bus_registry().
        Multiple calls are safe (handlers are overwritten).
    """
    handler_registry = get_handler_registry()
    event_bus_registry = get_event_bus_registry()

    # Register all known handlers
    for handler_type, (handler_cls, description) in _KNOWN_HANDLERS.items():
        # Note: Handlers implement ProtocolHandler structurally but don't inherit from it
        handler_registry.register(handler_type, handler_cls)  # type: ignore[arg-type]
        logger.debug(
            "Registered handler",
            extra={
                "handler_type": handler_type,
                "handler_class": handler_cls.__name__,
                "description": description,
            },
        )

    # Register all known event buses
    # Note: EventBusBindingRegistry raises if already registered,
    # so we check first to make this idempotent
    for bus_kind, (bus_cls, description) in _KNOWN_EVENT_BUSES.items():
        if not event_bus_registry.is_registered(bus_kind):
            event_bus_registry.register(bus_kind, bus_cls)
            logger.debug(
                "Registered event bus",
                extra={
                    "bus_kind": bus_kind,
                    "bus_class": bus_cls.__name__,
                    "description": description,
                },
            )

    registered_handlers = handler_registry.list_protocols()
    registered_buses = event_bus_registry.list_bus_kinds()

    logger.info(
        "Default handlers wired",
        extra={
            "handler_count": len(registered_handlers),
            "handlers": registered_handlers,
            "event_bus_count": len(registered_buses),
            "event_buses": registered_buses,
        },
    )

    return {
        "handlers": registered_handlers,
        "event_buses": registered_buses,
    }


def wire_handlers_from_contract(
    contract_config: dict[str, object],
) -> dict[str, list[str]]:
    """Register handlers and event buses based on contract configuration.

    This function selectively registers handlers based on the configuration
    provided in a contract. It validates that all requested handler types
    are known and supported.

    Args:
        contract_config: Contract configuration dict containing:
            - handlers: Optional list of handler configs, each with:
                - type: Handler type string (e.g., "http", "db")
                - enabled: Optional bool, defaults to True
            - event_bus: Optional event bus config with:
                - kind: Event bus kind string (e.g., "inmemory")
                - enabled: Optional bool, defaults to True

    Returns:
        Summary dict with keys:
            - handlers: List of registered handler type names
            - event_buses: List of registered event bus kind names

    Raises:
        ProtocolConfigurationError: If an unknown handler type or event bus
            kind is specified in the configuration.

    Example:
        >>> config = {
        ...     "handlers": [
        ...         {"type": "http", "enabled": True},
        ...         {"type": "db", "enabled": False},  # Skipped
        ...     ],
        ...     "event_bus": {"kind": "inmemory"},
        ... }
        >>> summary = wire_handlers_from_contract(config)
        >>> print(summary)
        {'handlers': ['http'], 'event_buses': ['inmemory']}

    Note:
        Disabled handlers (enabled=False) are skipped during registration.
        Unknown handler types raise ProtocolConfigurationError immediately.
    """
    handler_registry = get_handler_registry()
    event_bus_registry = get_event_bus_registry()

    registered_handlers: list[str] = []
    registered_buses: list[str] = []

    # Process handler configurations
    handlers_config = contract_config.get("handlers")
    if handlers_config is not None:
        if not isinstance(handlers_config, list):
            raise ProtocolConfigurationError(
                "Contract 'handlers' must be a list of handler configurations"
            )

        for handler_config in handlers_config:
            if not isinstance(handler_config, dict):
                raise ProtocolConfigurationError(
                    "Each handler configuration must be a dict"
                )

            handler_type = handler_config.get("type")
            if not isinstance(handler_type, str):
                raise ProtocolConfigurationError(
                    "Handler configuration missing required 'type' field"
                )

            # Check if handler is enabled (default True)
            enabled = handler_config.get("enabled", True)
            if not enabled:
                logger.debug(
                    "Skipping disabled handler",
                    extra={"handler_type": handler_type},
                )
                continue

            # Validate handler type is known
            if handler_type not in _KNOWN_HANDLERS:
                known_types = sorted(_KNOWN_HANDLERS.keys())
                raise ProtocolConfigurationError(
                    f"Unknown handler type: {handler_type!r}. "
                    f"Known types: {known_types}"
                )

            # Register the handler
            handler_cls, description = _KNOWN_HANDLERS[handler_type]
            # Note: Handlers implement ProtocolHandler structurally but don't inherit from it
            handler_registry.register(handler_type, handler_cls)  # type: ignore[arg-type]
            registered_handlers.append(handler_type)

            logger.debug(
                "Registered handler from contract",
                extra={
                    "handler_type": handler_type,
                    "handler_class": handler_cls.__name__,
                    "description": description,
                },
            )

    # Process event bus configuration
    event_bus_config = contract_config.get("event_bus")
    if event_bus_config is not None:
        if not isinstance(event_bus_config, dict):
            raise ProtocolConfigurationError(
                "Contract 'event_bus' must be a configuration dict"
            )

        bus_kind = event_bus_config.get("kind")
        if not isinstance(bus_kind, str):
            raise ProtocolConfigurationError(
                "Event bus configuration missing required 'kind' field"
            )

        # Check if event bus is enabled (default True)
        enabled = event_bus_config.get("enabled", True)
        if not enabled:
            logger.debug(
                "Skipping disabled event bus",
                extra={"bus_kind": bus_kind},
            )
        else:
            # Validate bus kind is known
            if bus_kind not in _KNOWN_EVENT_BUSES:
                known_kinds = sorted(_KNOWN_EVENT_BUSES.keys())
                raise ProtocolConfigurationError(
                    f"Unknown event bus kind: {bus_kind!r}. Known kinds: {known_kinds}"
                )

            # Register the event bus (check if already registered first)
            bus_cls, description = _KNOWN_EVENT_BUSES[bus_kind]
            if not event_bus_registry.is_registered(bus_kind):
                event_bus_registry.register(bus_kind, bus_cls)

            registered_buses.append(bus_kind)

            logger.debug(
                "Registered event bus from contract",
                extra={
                    "bus_kind": bus_kind,
                    "bus_class": bus_cls.__name__,
                    "description": description,
                },
            )

    logger.info(
        "Handlers wired from contract",
        extra={
            "handler_count": len(registered_handlers),
            "handlers": sorted(registered_handlers),
            "event_bus_count": len(registered_buses),
            "event_buses": sorted(registered_buses),
        },
    )

    return {
        "handlers": sorted(registered_handlers),
        "event_buses": sorted(registered_buses),
    }


def get_known_handler_types() -> list[str]:
    """Get list of known handler types that can be wired.

    Returns:
        Sorted list of handler type strings.

    Example:
        >>> get_known_handler_types()
        ['db', 'http']
    """
    return sorted(_KNOWN_HANDLERS.keys())


def get_known_event_bus_kinds() -> list[str]:
    """Get list of known event bus kinds that can be wired.

    Returns:
        Sorted list of event bus kind strings.

    Example:
        >>> get_known_event_bus_kinds()
        ['inmemory']
    """
    return sorted(_KNOWN_EVENT_BUSES.keys())


def wire_custom_handler(
    handler_type: str,
    handler_cls: type[ProtocolHandler],
    registry: ProtocolBindingRegistry | None = None,
) -> None:
    """Register a custom handler class with the registry.

    This function allows registration of custom handler implementations
    that are not part of the default handler set. Useful for testing
    and extending the handler ecosystem.

    Args:
        handler_type: Protocol type identifier for the handler.
        handler_cls: Handler class implementing ProtocolHandler protocol.
        registry: Optional registry to use. Defaults to singleton registry.

    Example:
        >>> class CustomHandler:
        ...     async def initialize(self, config): pass
        ...     async def execute(self, envelope): pass
        ...
        >>> wire_custom_handler("custom", CustomHandler)
    """
    target_registry = registry if registry is not None else get_handler_registry()
    target_registry.register(handler_type, handler_cls)
    logger.debug(
        "Registered custom handler",
        extra={
            "handler_type": handler_type,
            "handler_class": handler_cls.__name__,
        },
    )


def wire_custom_event_bus(
    bus_kind: str,
    bus_cls: type[ProtocolEventBus],
    registry: EventBusBindingRegistry | None = None,
) -> None:
    """Register a custom event bus class with the registry.

    This function allows registration of custom event bus implementations
    that are not part of the default set. Useful for testing and
    extending the event bus ecosystem.

    Args:
        bus_kind: Unique identifier for the bus type.
        bus_cls: Event bus class implementing ProtocolEventBus protocol.
        registry: Optional registry to use. Defaults to singleton registry.

    Raises:
        RuntimeHostError: If bus_kind is already registered in the registry.

    Example:
        >>> class CustomEventBus:
        ...     async def start(self): pass
        ...     async def publish(self, topic, key, value): pass
        ...
        >>> wire_custom_event_bus("custom", CustomEventBus)
    """
    target_registry = registry if registry is not None else get_event_bus_registry()
    # Note: EventBusBindingRegistry.register() raises if already registered
    target_registry.register(bus_kind, bus_cls)
    logger.debug(
        "Registered custom event bus",
        extra={
            "bus_kind": bus_kind,
            "bus_class": bus_cls.__name__,
        },
    )


__all__: list[str] = [
    # Primary wiring functions
    "wire_default_handlers",
    "wire_handlers_from_contract",
    # Introspection functions
    "get_known_handler_types",
    "get_known_event_bus_kinds",
    # Custom registration functions
    "wire_custom_handler",
    "wire_custom_event_bus",
]
