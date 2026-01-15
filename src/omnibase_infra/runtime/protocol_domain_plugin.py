# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Domain plugin protocol for kernel-level initialization hooks.

This module defines the ProtocolDomainPlugin protocol, enabling domain-specific
initialization to be decoupled from the generic runtime kernel. Domains (such as
Registration, Intelligence, etc.) can implement this protocol to hook into the
kernel bootstrap sequence.

Design Pattern:
    The plugin pattern follows dependency inversion - the kernel depends on the
    abstract ProtocolDomainPlugin protocol, not concrete implementations. Each
    domain provides its own plugin that implements the protocol.

    ```
    +-------------------------------------------------------------+
    |                        Kernel Layer                         |
    |  +--------------------------------------------------------+ |
    |  |  kernel.py                                             | |
    |  |    - Discovers plugins via registry                    | |
    |  |    - Calls plugin hooks during bootstrap               | |
    |  |    - NO domain-specific code                           | |
    |  +--------------------------------------------------------+ |
    |                            |                                 |
    |                            v                                 |
    |  +--------------------------------------------------------+ |
    |  |  ProtocolDomainPlugin (this file)                      | |
    |  |    - Defines initialization hooks                      | |
    |  |    - Plugin identification (plugin_id)                 | |
    |  |    - Lifecycle hooks (initialize, wire_handlers, etc.) | |
    |  +--------------------------------------------------------+ |
    +-------------------------------------------------------------+
                                 |
              +------------------+------------------+
              v                  v                  v
    +-----------------+ +-----------------+ +-----------------+
    |  Registration   | |  Intelligence   | |  Future Domain  |
    |  Plugin         | |  Plugin         | |  Plugin         |
    +-----------------+ +-----------------+ +-----------------+
    ```

Lifecycle Hooks:
    Plugins are initialized in a specific order during kernel bootstrap:

    1. `should_activate()` - Check if plugin should activate based on environment
    2. `initialize()` - Create domain-specific resources (pools, connections)
    3. `wire_handlers()` - Register handlers in the container
    4. `wire_dispatchers()` - Register dispatchers with MessageDispatchEngine
    5. `start_consumers()` - Start event consumers
    6. `shutdown()` - Clean up resources during kernel shutdown

Plugin Discovery:
    Plugins can be registered explicitly via `PluginDomainRegistry.register()`.
    This explicit registration approach provides:
    - Clear, auditable plugin loading
    - No magic discovery that could load untrusted code
    - Easy testing with mock plugins

Example Implementation:
    ```python
    from omnibase_infra.runtime.protocol_domain_plugin import ProtocolDomainPlugin
    from omnibase_infra.runtime.models import (
        ModelDomainPluginConfig,
        ModelDomainPluginResult,
    )

    class PluginMyDomain:
        '''Domain plugin for MyDomain.'''

        @property
        def plugin_id(self) -> str:
            return "my-domain"

        @property
        def display_name(self) -> str:
            return "My Domain"

        def should_activate(self, config: ModelDomainPluginConfig) -> bool:
            return bool(os.getenv("MY_DOMAIN_HOST"))

        async def initialize(
            self,
            config: ModelDomainPluginConfig,
        ) -> ModelDomainPluginResult:
            # Create pools, connections, etc.
            self._pool = await create_pool()
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                resources_created=["pool"],
            )

        async def wire_handlers(
            self,
            config: ModelDomainPluginConfig,
        ) -> ModelDomainPluginResult:
            # Register handlers with container
            await wire_my_domain_handlers(config.container, self._pool)
            return ModelDomainPluginResult.succeeded(
                plugin_id=self.plugin_id,
                services_registered=["MyHandler"],
            )

        async def shutdown(
            self,
            config: ModelDomainPluginConfig,
        ) -> ModelDomainPluginResult:
            await self._pool.close()
            return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)
    ```

Related:
    - OMN-1346: Registration Code Extraction
    - OMN-888: Registration Orchestrator
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from omnibase_infra.runtime.models import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ProtocolDomainPlugin(Protocol):
    """Protocol for domain-specific initialization plugins.

    Domain plugins implement this protocol to hook into the kernel bootstrap
    sequence. Each plugin is responsible for initializing its domain-specific
    resources, wiring handlers, and cleaning up during shutdown.

    The protocol uses duck typing - any class that implements these methods
    can be used as a domain plugin without explicit inheritance.

    Thread Safety:
        Plugin implementations should be thread-safe if they maintain state.
        The kernel calls plugin methods sequentially during bootstrap, but
        plugins may be accessed concurrently during runtime.

    Lifecycle Order:
        1. should_activate() - Check environment/config
        2. initialize() - Create pools, connections
        3. wire_handlers() - Register handlers in container
        4. wire_dispatchers() - Register with dispatch engine (optional)
        5. start_consumers() - Start event consumers (optional)
        6. shutdown() - Clean up during kernel shutdown

    Example:
        ```python
        class PluginMyDomain:
            @property
            def plugin_id(self) -> str:
                return "my-domain"

            def should_activate(self, config: ModelDomainPluginConfig) -> bool:
                return bool(os.getenv("MY_DOMAIN_ENABLED"))

            async def initialize(
                self, config: ModelDomainPluginConfig
            ) -> ModelDomainPluginResult:
                # Initialize domain resources
                return ModelDomainPluginResult.succeeded("my-domain")

            # ... other methods
        ```
    """

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin.

        The plugin_id is used for:
        - Logging and diagnostics
        - Plugin registry lookups
        - Status reporting in kernel banner

        Returns:
            Unique string identifier (e.g., "registration", "intelligence").
        """
        ...

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin.

        Used in logs and user-facing output.

        Returns:
            Display name (e.g., "Registration", "Intelligence").
        """
        ...

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Check if this plugin should activate based on configuration.

        Called during bootstrap to determine if the plugin should run.
        Plugins can check environment variables, config values, or other
        conditions to decide whether to activate.

        Args:
            config: Plugin configuration with container and event bus.

        Returns:
            True if the plugin should activate, False to skip.

        Example:
            ```python
            def should_activate(self, config: ModelDomainPluginConfig) -> bool:
                # Only activate if PostgreSQL is configured
                return bool(os.getenv("POSTGRES_HOST"))
            ```
        """
        ...

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Initialize domain-specific resources.

        Called after should_activate() returns True. This method should
        create any resources the domain needs (database pools, connections,
        etc.).

        Args:
            config: Plugin configuration with container and event bus.

        Returns:
            Result indicating success/failure and resources created.

        Example:
            ```python
            async def initialize(
                self, config: ModelDomainPluginConfig
            ) -> ModelDomainPluginResult:
                self._pool = await asyncpg.create_pool(dsn)
                return ModelDomainPluginResult.succeeded(
                    "my-domain",
                    resources_created=["postgres_pool"],
                )
            ```
        """
        ...

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register handlers with the container.

        Called after initialize(). This method should register any
        handlers the domain provides in the container's service registry.

        Args:
            config: Plugin configuration with container and event bus.

        Returns:
            Result indicating success/failure and services registered.

        Example:
            ```python
            async def wire_handlers(
                self, config: ModelDomainPluginConfig
            ) -> ModelDomainPluginResult:
                summary = await wire_my_handlers(config.container, self._pool)
                return ModelDomainPluginResult.succeeded(
                    "my-domain",
                    services_registered=summary["services"],
                )
            ```
        """
        ...

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register dispatchers with MessageDispatchEngine (optional).

        Called after wire_handlers(). This method should register any
        dispatchers the domain provides with the dispatch engine.

        Note: config.dispatch_engine may be None if no engine is configured.
        Implementations should handle this gracefully.

        Args:
            config: Plugin configuration with dispatch_engine set.

        Returns:
            Result indicating success/failure and dispatchers registered.
        """
        ...

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start event consumers (optional).

        Called after wire_dispatchers(). This method should start any
        event consumers the domain needs to process events from the bus.

        Returns:
            Result with unsubscribe_callbacks for cleanup during shutdown.
        """
        ...

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Clean up domain resources during kernel shutdown.

        Called during kernel shutdown. This method should close pools,
        connections, and any other resources created during initialize().

        Shutdown Order (LIFO):
            Plugins are shut down in **reverse activation order** (Last In, First Out).
            This ensures plugins activated later are shut down before plugins they may
            depend on. For example, if plugins A, B, C are activated in order, shutdown
            order is C, B, A.

        Self-Contained Constraint:
            **CRITICAL**: Plugins MUST be self-contained during shutdown.

            - Plugins MUST NOT depend on resources from other plugins during shutdown
            - Each plugin should only clean up its own resources (pools, connections)
            - If a plugin accesses shared resources, it must handle graceful degradation
              in case those resources are already released by another plugin
            - Shutdown errors in one plugin do not block other plugins from shutting down

            This constraint exists because:
            1. Shutdown order may change as plugins are added/removed
            2. Other plugins may fail to initialize, leaving resources unavailable
            3. Exception handling during shutdown should not cascade failures

        Error Handling:
            Implementations should catch and log errors rather than raising them.
            The kernel will continue shutting down other plugins even if one fails.
            Return a failed ModelDomainPluginResult to report errors without blocking.

        Args:
            config: Plugin configuration. Note that during cleanup after errors,
                a minimal config may be passed instead of the original config.

        Returns:
            Result indicating success/failure of cleanup.

        Example:
            ```python
            async def shutdown(
                self, config: ModelDomainPluginConfig
            ) -> ModelDomainPluginResult:
                errors: list[str] = []

                # Close pool - handle graceful degradation
                if self._pool is not None:
                    try:
                        await self._pool.close()
                    except Exception as e:
                        errors.append(f"pool: {e}")
                    self._pool = None  # Always clear reference

                if errors:
                    return ModelDomainPluginResult.failed(
                        plugin_id=self.plugin_id,
                        error_message="; ".join(errors),
                    )
                return ModelDomainPluginResult.succeeded(plugin_id=self.plugin_id)
            ```
        """
        ...


class RegistryDomainPlugin:
    """Registry for domain plugins.

    Provides explicit plugin registration and discovery. Plugins must be
    registered explicitly - there is no automatic discovery to prevent
    loading untrusted code.

    Thread Safety:
        The registry is NOT thread-safe. Plugin registration should happen
        during startup before concurrent access.

    Example:
        ```python
        from omnibase_infra.runtime.protocol_domain_plugin import (
            RegistryDomainPlugin,
        )
        from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
            PluginRegistration,
        )

        # Register plugins explicitly
        registry = RegistryDomainPlugin()
        registry.register(PluginRegistration())

        # Get all registered plugins
        plugins = registry.get_all()
        for plugin in plugins:
            if plugin.should_activate(config):
                await plugin.initialize(config)
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty plugin registry."""
        self._plugins: dict[str, ProtocolDomainPlugin] = {}

    def register(self, plugin: ProtocolDomainPlugin) -> None:
        """Register a domain plugin.

        Args:
            plugin: Plugin instance implementing ProtocolDomainPlugin.

        Raises:
            ValueError: If a plugin with the same ID is already registered.
        """
        plugin_id = plugin.plugin_id
        if plugin_id in self._plugins:
            raise ValueError(
                f"Plugin with ID '{plugin_id}' is already registered. "
                f"Each plugin must have a unique plugin_id."
            )
        self._plugins[plugin_id] = plugin
        logger.debug(
            "Registered domain plugin",
            extra={
                "plugin_id": plugin_id,
                "display_name": plugin.display_name,
            },
        )

    def get(self, plugin_id: str) -> ProtocolDomainPlugin | None:
        """Get a plugin by ID.

        Args:
            plugin_id: The plugin identifier.

        Returns:
            The plugin instance, or None if not found.
        """
        return self._plugins.get(plugin_id)

    def get_all(self) -> list[ProtocolDomainPlugin]:
        """Get all registered plugins.

        Returns:
            List of all registered plugin instances.
        """
        return list(self._plugins.values())

    def clear(self) -> None:
        """Clear all registered plugins (useful for testing)."""
        self._plugins.clear()

    def __len__(self) -> int:
        """Return number of registered plugins."""
        return len(self._plugins)


__all__: list[str] = [
    "ModelDomainPluginConfig",
    "ModelDomainPluginResult",
    "ProtocolDomainPlugin",
    "RegistryDomainPlugin",
]
