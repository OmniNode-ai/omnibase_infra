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
    Plugins support hybrid discovery with two mechanisms:

    1. **Explicit registration** via ``RegistryDomainPlugin.register()`` --
       the primary path for first-party plugins. Provides clear, auditable
       loading and easy testing with mock plugins.

    2. **Entry-point discovery** via ``RegistryDomainPlugin.discover_from_entry_points()``
       -- secondary mechanism for external packages. Uses PEP 621 entry_points
       (``importlib.metadata``) to scan installed packages for plugins
       declared under the ``onex.domain_plugins`` group.

    Explicit registration always takes precedence on duplicate ``plugin_id``.
    Entry-point discovery is security-gated by namespace allowlisting
    (pre-import) and protocol validation (post-import).

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
from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from omnibase_infra.runtime.constants_security import (
    DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
    TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
)
from omnibase_infra.runtime.models import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
)
from omnibase_infra.runtime.models.model_plugin_discovery_entry import (
    ModelPluginDiscoveryEntry,
)
from omnibase_infra.runtime.models.model_plugin_discovery_report import (
    ModelPluginDiscoveryReport,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

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
    """Registry for domain plugins with hybrid discovery.

    Provides two discovery mechanisms:

    1. **Explicit registration** via ``register()`` -- the primary path for
       first-party plugins. Clear, auditable, and easy to test.

    2. **Entry-point discovery** via ``discover_from_entry_points()`` --
       secondary mechanism for external packages. Uses Python entry_points
       (PEP 621) to scan installed packages. Security-gated by namespace
       allowlisting and protocol validation.

    Explicit registration always wins on duplicate ``plugin_id``.

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

        # 1. Register first-party plugins explicitly
        registry = RegistryDomainPlugin()
        registry.register(PluginRegistration())

        # 2. Discover third-party plugins from entry_points
        report = registry.discover_from_entry_points()
        # Explicit registration wins on duplicate plugin_id

        # 3. Activate all registered plugins
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

    def discover_from_entry_points(
        self,
        group: str = DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
        allowed_namespaces: tuple[str, ...] = TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
    ) -> ModelPluginDiscoveryReport:
        """Discover and register domain plugins from Python entry_points.

        Scans the given entry_point group for plugin classes, validates them
        against the namespace allowlist and ProtocolDomainPlugin protocol,
        then registers any that pass. Plugins already registered via explicit
        ``register()`` calls are skipped (explicit wins on duplicate plugin_id).

        Security Model:
            1. **Pre-import namespace validation** -- ``entry_point.value``
               (the dotted module path) is checked against ``allowed_namespaces``
               BEFORE ``entry_point.load()`` is called. No code is executed for
               rejected namespaces.
            2. **Post-import protocol validation** -- After loading the class and
               instantiating it, ``isinstance(instance, ProtocolDomainPlugin)``
               validates structural conformance.
            3. **Duplicate-safe** -- If a plugin with the same ``plugin_id`` was
               already registered (e.g. via explicit ``register()``), the
               entry_point is silently skipped with status ``"duplicate_skipped"``.

        Args:
            group: PEP 621 entry_point group name to scan.
                Defaults to ``DOMAIN_PLUGIN_ENTRY_POINT_GROUP``
                (``"onex.domain_plugins"``).
            allowed_namespaces: Tuple of namespace prefixes that are trusted
                for dynamic import. Only entry_points whose ``value`` starts
                with one of these prefixes will be loaded.
                Defaults to ``TRUSTED_PLUGIN_NAMESPACE_PREFIXES``.

        Returns:
            ModelPluginDiscoveryReport with per-entry diagnostics.

        Example:
            >>> registry = RegistryDomainPlugin()
            >>> report = registry.discover_from_entry_points()
            >>> for entry in report.entries:
            ...     print(f"{entry.entry_point_name}: {entry.status}")
        """
        discovered_eps = entry_points(group=group)
        entries: list[ModelPluginDiscoveryEntry] = []
        accepted: list[str] = []

        for ep in discovered_eps:
            ep_name = ep.name
            # entry_point.value is the "module:class" string, e.g.
            # "omnibase_infra.nodes...plugin:PluginRegistration"
            module_path = ep.value

            # --- Pre-import namespace validation ---
            if not any(module_path.startswith(prefix) for prefix in allowed_namespaces):
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep_name,
                        module_path=module_path,
                        status="namespace_rejected",
                        reason=(
                            f"Module path '{module_path}' does not start with "
                            f"any allowed namespace prefix: {allowed_namespaces}"
                        ),
                    )
                )
                logger.info(
                    "Plugin entry_point '%s' rejected: namespace '%s' not in "
                    "allowed prefixes",
                    ep_name,
                    module_path,
                )
                continue

            # --- Load the class (executes module-level code) ---
            try:
                plugin_class = ep.load()
            except Exception as load_err:
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep_name,
                        module_path=module_path,
                        status="import_error",
                        reason=sanitize_error_message(load_err),
                    )
                )
                logger.warning(
                    "Plugin entry_point '%s' failed to load: %s",
                    ep_name,
                    sanitize_error_message(load_err),
                )
                continue

            # --- Instantiate the plugin ---
            try:
                instance = plugin_class()
            except Exception as init_err:
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep_name,
                        module_path=module_path,
                        status="instantiation_error",
                        reason=sanitize_error_message(init_err),
                    )
                )
                logger.warning(
                    "Plugin entry_point '%s' failed to instantiate: %s",
                    ep_name,
                    sanitize_error_message(init_err),
                )
                continue

            # --- Protocol validation ---
            if not isinstance(instance, ProtocolDomainPlugin):
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep_name,
                        module_path=module_path,
                        status="protocol_invalid",
                        reason=(
                            f"Class '{type(instance).__name__}' does not implement "
                            f"ProtocolDomainPlugin protocol"
                        ),
                    )
                )
                logger.warning(
                    "Plugin entry_point '%s' does not implement "
                    "ProtocolDomainPlugin: %s",
                    ep_name,
                    type(instance).__name__,
                )
                continue

            # --- Duplicate check (explicit wins) ---
            plugin_id = instance.plugin_id
            if plugin_id in self._plugins:
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep_name,
                        module_path=module_path,
                        status="duplicate_skipped",
                        reason=(
                            f"Plugin ID '{plugin_id}' already registered "
                            f"(explicit registration takes precedence)"
                        ),
                        plugin_id=plugin_id,
                    )
                )
                logger.debug(
                    "Plugin entry_point '%s' skipped: plugin_id '%s' already "
                    "registered (explicit registration wins)",
                    ep_name,
                    plugin_id,
                )
                continue

            # --- Register ---
            self._plugins[plugin_id] = instance
            accepted.append(plugin_id)
            entries.append(
                ModelPluginDiscoveryEntry(
                    entry_point_name=ep_name,
                    module_path=module_path,
                    status="accepted",
                    plugin_id=plugin_id,
                )
            )
            logger.info(
                "Plugin entry_point '%s' discovered and registered "
                "(plugin_id='%s', module='%s')",
                ep_name,
                plugin_id,
                module_path,
            )

        report = ModelPluginDiscoveryReport(
            group=group,
            discovered_count=len(list(discovered_eps)),
            accepted=tuple(accepted),
            entries=tuple(entries),
        )

        logger.info(
            "Plugin discovery for group '%s': %d discovered, %d accepted, %d rejected",
            group,
            report.discovered_count,
            len(report.accepted),
            len(report.rejected),
        )

        return report


__all__: list[str] = [
    "ModelDomainPluginConfig",
    "ModelDomainPluginResult",
    "ProtocolDomainPlugin",
    "RegistryDomainPlugin",
]
