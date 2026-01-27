# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Wiring Module - Registers concrete handlers with the handler registry.

This module provides functions to wire up concrete handler implementations
with the RegistryProtocolBinding and RegistryEventBusBinding. It serves as
the bridge between handler implementations and the registry system.

The wiring module is responsible for:
- Registering default handlers from contract.yaml files
- Registering handlers based on contract configuration
- Validating that requested handler types are known and supported
- Providing a summary of registered handlers for debugging

Contract-Driven Handler Loading:
    Handler classes are discovered and loaded from contract.yaml files located
    in nodes/handlers/<handler_type>/contract.yaml. Each contract specifies:
    - handler.module: The Python module path
    - handler.name: The class name to load

    This replaces the old hardcoded _KNOWN_HANDLERS dict with dynamic,
    contract-based discovery.

Event Bus Support:
    This module registers EventBusInmemory as the default event bus. For production
    deployments requiring EventBusKafka, the event bus is selected at kernel bootstrap
    time based on:
    - KAFKA_BOOTSTRAP_SERVERS environment variable (if set, uses EventBusKafka)
    - config.event_bus.type field in runtime_config.yaml

    See kernel.py for event bus selection logic during runtime bootstrap.

Design Principles:
- Contract-driven: Handler configurations live in contract.yaml, not Python code
- Explicit wiring: All handler registrations are explicit, not auto-discovered
- Validation: Unknown handler types raise clear errors
- Fail-fast: Missing contracts raise FileNotFoundError immediately
- Idempotent: Re-wiring the same handler is safe (overwrites previous)
- Security: Namespace allowlisting is recommended for production deployments
  (see docs/patterns/handler_plugin_loader.md#optional-security-controls)

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

    2. Create a contract.yaml in nodes/handlers/<type>/contract.yaml:

        ```yaml
        name: "handler_custom"
        node_type: "EFFECT_GENERIC"
        description: "Custom protocol handler"
        handler_routing:
          routing_strategy: "operation_match"
          handlers:
            - handler_type: "custom"
              handler:
                name: "MyCustomHandler"
                module: "mypackage.handlers.handler_custom"
        ```

    3. Add the contract path to _HANDLER_CONTRACT_PATHS in this module.

    4. For runtime registration without contracts, use wire_custom_handler():

        ```python
        from omnibase_infra.runtime.util_wiring import wire_custom_handler

        wire_custom_handler("custom", MyCustomHandler)
        ```

Integration with RuntimeHostProcess:
    The wiring module and RuntimeHostProcess work together in a two-phase
    registration and instantiation flow:

    Phase 1 - Class Registration (wiring module):
        When wire_default_handlers() is called, handler CLASSES are registered
        with the singleton RegistryProtocolBinding. At this point, no handler
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
    from omnibase_infra.runtime.util_wiring import (
        wire_default_handlers,
        wire_handlers_from_contract,
    )

    # Wire all default handlers from contracts
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

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_core.types import JsonType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.handler_registry import (
    EVENT_BUS_INMEMORY,
    HANDLER_TYPE_CONSUL,
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_GRAPH,
    HANDLER_TYPE_HTTP,
    HANDLER_TYPE_INTENT,
    HANDLER_TYPE_MCP,
    HANDLER_TYPE_VAULT,
    RegistryEventBusBinding,
    RegistryProtocolBinding,
    get_event_bus_registry,
    get_handler_registry,
)

if TYPE_CHECKING:
    from omnibase_core.protocol.protocol_event_bus import ProtocolEventBus
    from omnibase_infra.protocols import ProtocolContainerAware

logger = logging.getLogger(__name__)

# Handler contract directory path.
# Handler configurations are loaded from contract.yaml files in this directory.
_HANDLERS_BASE = Path(__file__).parent.parent / "nodes" / "handlers"

# Mapping of handler types to their contract paths.
# Each entry maps a handler type constant to the path of its contract.yaml file.
_HANDLER_CONTRACT_PATHS: dict[str, Path] = {
    HANDLER_TYPE_CONSUL: _HANDLERS_BASE / "consul" / "contract.yaml",
    HANDLER_TYPE_DATABASE: _HANDLERS_BASE / "db" / "contract.yaml",
    HANDLER_TYPE_GRAPH: _HANDLERS_BASE / "graph" / "contract.yaml",
    HANDLER_TYPE_HTTP: _HANDLERS_BASE / "http" / "contract.yaml",
    HANDLER_TYPE_INTENT: _HANDLERS_BASE / "intent" / "contract.yaml",
    HANDLER_TYPE_MCP: _HANDLERS_BASE / "mcp" / "contract.yaml",
    HANDLER_TYPE_VAULT: _HANDLERS_BASE / "vault" / "contract.yaml",
}

# Known event bus kinds that can be wired via this module.
# Maps bus kind constant to (bus_class, description)
#
# Note: EventBusKafka is NOT in this registry. It is selected at kernel bootstrap
# time via environment variable (KAFKA_BOOTSTRAP_SERVERS) or runtime config
# (event_bus.type = "kafka"). This registry handles only contract-based wiring
# while production event bus selection is handled by kernel.py.
_KNOWN_EVENT_BUSES: dict[str, tuple[type[ProtocolEventBus], str]] = {
    EVENT_BUS_INMEMORY: (EventBusInmemory, "In-memory event bus for local/testing"),
}


def _load_handler_from_contract(
    handler_type: str, contract_path: Path
) -> tuple[type[ProtocolContainerAware], str]:
    """Load handler class from a contract.yaml file.

    Args:
        handler_type: The handler type identifier (e.g., "consul", "db").
        contract_path: Path to the contract.yaml file.

    Returns:
        Tuple of (handler_class, description).

    Raises:
        FileNotFoundError: If contract file does not exist.
        ProtocolConfigurationError: If contract is malformed or handler cannot be loaded.

    Security Note:
        This function uses ``importlib.import_module()`` to dynamically load handler
        modules specified in contracts. This means contract files are effectively
        executable code - a compromised contract pointing to a malicious module
        will execute that module's code during import.

        **Production Security Recommendations:**

        1. **Namespace Allowlisting**: For dynamic handler discovery scenarios,
           use ``HandlerPluginLoader`` with the ``allowed_namespaces`` parameter
           to restrict which module namespaces can be loaded::

               from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

               loader = HandlerPluginLoader(
                   allowed_namespaces=["omnibase_infra.", "omnibase_core.", "myapp.handlers."]
               )

           This prevents loading handlers from untrusted namespaces even if a
           contract is compromised.

        2. **Write Protection**: Contract directories should be read-only at runtime.
           Mount contract directories as read-only volumes in containerized deployments.

        3. **Source Validation**: Contracts in ``_HANDLER_CONTRACT_PATHS`` come from
           the omnibase_infra package. Ensure these are from trusted, version-controlled
           sources with code review.

        See Also:
            - ``docs/patterns/handler_plugin_loader.md#optional-security-controls``
            - ``docs/patterns/security_patterns.md``
            - ``docs/decisions/adr-handler-plugin-loader-security.md``
    """
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Handler contract not found: {contract_path}. "
            f"All handlers must have contract.yaml files."
        )

    with contract_path.open("r") as f:
        contract = yaml.safe_load(f)

    if contract is None:
        context = ModelInfraErrorContext.with_correlation(
            operation="load_handler_contract",
            target_name=str(contract_path),
        )
        raise ProtocolConfigurationError(
            f"Empty contract file: {contract_path}",
            context=context,
        )

    handler_routing = contract.get("handler_routing", {})
    handlers = handler_routing.get("handlers", [])

    if not handlers:
        context = ModelInfraErrorContext.with_correlation(
            operation="load_handler_contract",
            target_name=str(contract_path),
        )
        raise ProtocolConfigurationError(
            f"No handlers defined in contract: {contract_path}",
            context=context,
        )

    handler_def = handlers[0]
    handler_info = handler_def.get("handler", {})
    handler_module = handler_info.get("module")
    handler_class_name = handler_info.get("name")

    if not handler_module or not handler_class_name:
        context = ModelInfraErrorContext.with_correlation(
            operation="load_handler_contract",
            target_name=str(contract_path),
        )
        raise ProtocolConfigurationError(
            f"Missing handler module or name in contract: {contract_path}. "
            f"Expected handler.module and handler.name fields.",
            context=context,
        )

    try:
        module = importlib.import_module(handler_module)
    except ImportError as e:
        context = ModelInfraErrorContext.with_correlation(
            operation="import_handler_module",
            target_name=handler_module,
        )
        raise ProtocolConfigurationError(
            f"Failed to import handler module '{handler_module}': {e}",
            context=context,
        ) from e

    try:
        handler_class = getattr(module, handler_class_name)
    except AttributeError as e:
        context = ModelInfraErrorContext.with_correlation(
            operation="load_handler_class",
            target_name=f"{handler_module}.{handler_class_name}",
        )
        raise ProtocolConfigurationError(
            f"Handler class '{handler_class_name}' not found in module '{handler_module}'",
            context=context,
        ) from e

    description = contract.get("description", f"{handler_type} handler")

    logger.debug(
        "Loaded handler from contract",
        extra={
            "handler_type": handler_type,
            "handler_class": handler_class_name,
            "handler_module": handler_module,
            "contract_path": str(contract_path),
        },
    )

    return handler_class, description


def wire_default_handlers() -> dict[str, list[str]]:
    """Register all default handlers and event buses with singleton registries.

    This function registers the standard set of handlers and event buses
    that are available for use by the RuntimeHostProcess. It is the primary
    way to initialize the handler ecosystem.

    Registered Handlers:
        - CONSUL: HandlerConsul for HashiCorp Consul service discovery
        - DB: HandlerDb for PostgreSQL database operations
        - GRAPH: HandlerGraph for graph database (Memgraph/Neo4j) operations
        - HTTP: HandlerHttpRest for HTTP/REST protocol operations
        - INTENT: HandlerIntent for intent storage and query (demo)
        - MCP: HandlerMCP for Model Context Protocol AI agent integration
        - VAULT: HandlerVault for HashiCorp Vault secret management

    Registered Event Buses:
        - INMEMORY: EventBusInmemory for local/testing deployments

    Event Bus Selection Note:
        This function only registers EventBusInmemory in the event bus registry.
        For production deployments with EventBusKafka:
        - Set KAFKA_BOOTSTRAP_SERVERS environment variable, OR
        - Configure event_bus.type = "kafka" in runtime_config.yaml

        EventBusKafka selection happens at kernel bootstrap time (see kernel.py),
        not through this registry-based wiring mechanism.

    Returns:
        Summary dict with keys:
            - handlers: List of registered handler type names
            - event_buses: List of registered event bus kind names

    Example:
        >>> summary = wire_default_handlers()
        >>> print(summary)
        {'handlers': ['consul', 'db', 'http', 'mcp', 'vault'], 'event_buses': ['inmemory']}

    Note:
        This function uses the singleton registries returned by
        get_handler_registry() and get_event_bus_registry().
        Multiple calls are safe (handlers are overwritten).
    """
    handler_registry = get_handler_registry()
    event_bus_registry = get_event_bus_registry()

    # Register all handlers from contracts
    for handler_type, contract_path in _HANDLER_CONTRACT_PATHS.items():
        handler_cls, description = _load_handler_from_contract(
            handler_type, contract_path
        )
        # NOTE: Handlers implement ProtocolHandler structurally but don't inherit from it.
        # Mypy cannot verify structural subtyping for registration argument.
        handler_registry.register(handler_type, handler_cls)  # type: ignore[arg-type]  # NOTE: structural subtyping
        logger.debug(
            "Registered handler from contract",
            extra={
                "handler_type": handler_type,
                "handler_class": handler_cls.__name__,
                "description": description,
                "contract_path": str(contract_path),
            },
        )

    # Register all known event buses
    # Note: RegistryEventBusBinding raises if already registered,
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
    contract_config: dict[str, JsonType],
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

    # Create error context for configuration errors
    def _make_error_context(
        operation: str, target_name: str = "wiring"
    ) -> ModelInfraErrorContext:
        """Create standardized error context for configuration errors."""
        return ModelInfraErrorContext.with_correlation(
            operation=operation,
            target_name=target_name,
        )

    # Process handler configurations
    handlers_config = contract_config.get("handlers")
    if handlers_config is not None:
        if not isinstance(handlers_config, list):
            raise ProtocolConfigurationError(
                "Contract 'handlers' must be a list of handler configurations",
                context=_make_error_context("validate_handlers_config"),
            )

        for handler_config in handlers_config:
            if not isinstance(handler_config, dict):
                raise ProtocolConfigurationError(
                    "Each handler configuration must be a dict",
                    context=_make_error_context("validate_handler_entry"),
                )

            handler_type = handler_config.get("type")
            if not isinstance(handler_type, str):
                raise ProtocolConfigurationError(
                    "Handler configuration missing required 'type' field",
                    context=_make_error_context("validate_handler_type"),
                )

            # Check if handler is enabled (default True)
            enabled = handler_config.get("enabled", True)
            if not enabled:
                logger.debug(
                    "Skipping disabled handler",
                    extra={"handler_type": handler_type},
                )
                continue

            # Validate handler type is known (has a contract)
            if handler_type not in _HANDLER_CONTRACT_PATHS:
                known_types = sorted(_HANDLER_CONTRACT_PATHS.keys())
                raise ProtocolConfigurationError(
                    f"Unknown handler type: {handler_type!r}. "
                    f"Known types: {known_types}",
                    context=_make_error_context("validate_handler_type", handler_type),
                )

            # Load and register the handler from contract
            contract_path = _HANDLER_CONTRACT_PATHS[handler_type]
            handler_cls, description = _load_handler_from_contract(
                handler_type, contract_path
            )
            # NOTE: Handlers implement ProtocolHandler structurally but don't inherit from it.
            # Mypy cannot verify structural subtyping for registration argument.
            handler_registry.register(handler_type, handler_cls)  # type: ignore[arg-type]  # NOTE: structural subtyping
            registered_handlers.append(handler_type)

            logger.debug(
                "Registered handler from contract config",
                extra={
                    "handler_type": handler_type,
                    "handler_class": handler_cls.__name__,
                    "description": description,
                    "contract_path": str(contract_path),
                },
            )

    # Process event bus configuration
    event_bus_config = contract_config.get("event_bus")
    if event_bus_config is not None:
        if not isinstance(event_bus_config, dict):
            raise ProtocolConfigurationError(
                "Contract 'event_bus' must be a configuration dict",
                context=_make_error_context("validate_event_bus_config"),
            )

        bus_kind = event_bus_config.get("kind")
        if not isinstance(bus_kind, str):
            raise ProtocolConfigurationError(
                "Event bus configuration missing required 'kind' field",
                context=_make_error_context("validate_event_bus_kind"),
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
                    f"Unknown event bus kind: {bus_kind!r}. Known kinds: {known_kinds}",
                    context=_make_error_context("validate_event_bus_kind", bus_kind),
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

    Handler types are discovered from contract.yaml files in nodes/handlers/.

    Returns:
        Sorted list of handler type strings.

    Example:
        >>> get_known_handler_types()
        ['consul', 'db', 'graph', 'http', 'intent', 'mcp', 'vault']
    """
    return sorted(_HANDLER_CONTRACT_PATHS.keys())


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
    handler_cls: type[ProtocolContainerAware],
    registry: RegistryProtocolBinding | None = None,
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
    registry: RegistryEventBusBinding | None = None,
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
    # Note: RegistryEventBusBinding.register() raises if already registered
    target_registry.register(bus_kind, bus_cls)
    logger.debug(
        "Registered custom event bus",
        extra={
            "bus_kind": bus_kind,
            "bus_class": bus_cls.__name__,
        },
    )


__all__: list[str] = [
    "get_known_event_bus_kinds",
    # Introspection functions
    "get_known_handler_types",
    "wire_custom_event_bus",
    # Custom registration functions
    "wire_custom_handler",
    # Primary wiring functions
    "wire_default_handlers",
    "wire_handlers_from_contract",
]
