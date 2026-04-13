# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Domain plugin protocol and registry for kernel-level initialization hooks.

ProtocolDomainPlugin, ModelDomainPluginConfig, and ModelDomainPluginResult
are defined in omnibase_spi and re-exported here for backwards compatibility.

RegistryDomainPlugin lives here (it has runtime dependencies on infra models).

Plugin Discovery:
    1. Explicit registration via RegistryDomainPlugin.register()
    2. Entry-point discovery via RegistryDomainPlugin.discover_from_entry_points()

Related:
    - OMN-1346: Registration Code Extraction
    - OMN-888: Registration Orchestrator
    - OMN-8550: Moved ProtocolDomainPlugin to omnibase_spi
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points

from omnibase_infra.runtime.constants_security import (
    DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
)
from omnibase_infra.runtime.models.model_handshake_result import (
    ModelHandshakeResult,
)
from omnibase_infra.runtime.models.model_plugin_discovery_entry import (
    ModelPluginDiscoveryEntry,
)
from omnibase_infra.runtime.models.model_plugin_discovery_report import (
    ModelPluginDiscoveryReport,
)
from omnibase_infra.runtime.models.model_security_config import ModelSecurityConfig
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

# Re-exported from omnibase_spi for backwards compatibility (OMN-8550)
from omnibase_spi.protocols.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
    ProtocolDomainPlugin,
)

logger = logging.getLogger(__name__)


class RegistryDomainPlugin:
    """Registry for domain plugins with hybrid explicit + entry-point discovery.

    Provides two complementary registration mechanisms:

    1. **Explicit registration** via ``register()`` -- the primary path for
       first-party plugins. Direct, auditable, and easy to test.

    2. **Entry-point discovery** via ``discover_from_entry_points()`` --
       secondary mechanism for external packages. Uses PEP 621 entry_points
       to scan installed packages. Security-gated by namespace allowlisting
       and protocol validation.

    Explicit registrations always take precedence: if ``discover_from_entry_points()``
    finds a plugin whose ``plugin_id`` matches an already-registered plugin, the
    discovered plugin is silently skipped and recorded as ``"duplicate_skipped"``
    in the discovery report.

    Thread Safety:
        The registry is NOT thread-safe. Plugin registration should happen
        during startup before concurrent access.
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
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_all(self) -> list[ProtocolDomainPlugin]:
        """Get all registered plugins."""
        return list(self._plugins.values())

    def clear(self) -> None:
        """Clear all registered plugins (useful for testing)."""
        self._plugins.clear()

    def __len__(self) -> int:
        """Return number of registered plugins."""
        return len(self._plugins)

    def discover_from_entry_points(
        self,
        security_config: ModelSecurityConfig | None = None,
        *,
        group: str = DOMAIN_PLUGIN_ENTRY_POINT_GROUP,
        strict: bool = False,
    ) -> ModelPluginDiscoveryReport:
        """Discover and register plugins from PEP 621 entry points.

        Args:
            security_config: Security configuration controlling which
                namespaces are trusted for plugin loading.
            group: Entry-point group name to scan.
            strict: When True, raise on the first import or instantiation error.

        Returns:
            A ModelPluginDiscoveryReport containing all discovery outcomes.

        Raises:
            ImportError: Only when strict=True and an entry point fails to load.
            TypeError: Only when strict=True and a loaded class fails to instantiate.
            RuntimeError: Only when strict=True and a loaded class does not satisfy
                ProtocolDomainPlugin.
        """
        if security_config is None:
            security_config = ModelSecurityConfig()

        allowed_namespaces = security_config.get_effective_plugin_namespaces()

        eps = entry_points(group=group)
        sorted_eps = sorted(eps, key=lambda ep: (ep.name, ep.value))

        entries: list[ModelPluginDiscoveryEntry] = []
        accepted: list[str] = []

        for ep in sorted_eps:
            module_path = self._parse_module_path(ep.value)

            if not self._validate_plugin_namespace(module_path, allowed_namespaces):
                logger.info(
                    "Plugin entry point namespace rejected: %s (module: %s)",
                    ep.name,
                    module_path,
                )
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep.name,
                        module_path=module_path,
                        status="namespace_rejected",
                        reason=(
                            f"Module '{module_path}' is not in any trusted "
                            f"namespace: {allowed_namespaces}"
                        ),
                    )
                )
                continue

            try:
                loaded_class = ep.load()
            except Exception as exc:
                msg = sanitize_error_message(exc)
                logger.warning(
                    "Failed to load plugin entry point '%s': %s",
                    ep.name,
                    msg,
                )
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep.name,
                        module_path=module_path,
                        status="import_error",
                        reason=msg,
                    )
                )
                if strict:
                    raise ImportError(
                        f"Failed to load plugin entry point '{ep.name}': {msg}"
                    ) from exc
                continue

            try:
                plugin = loaded_class()
            except Exception as exc:
                msg = sanitize_error_message(exc)
                logger.warning(
                    "Failed to instantiate plugin from entry point '%s': %s",
                    ep.name,
                    msg,
                )
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep.name,
                        module_path=module_path,
                        status="instantiation_error",
                        reason=msg,
                    )
                )
                if strict:
                    raise TypeError(
                        f"Failed to instantiate plugin from entry point "
                        f"'{ep.name}': {msg}"
                    ) from exc
                continue

            if not isinstance(plugin, ProtocolDomainPlugin):
                class_name = getattr(loaded_class, "__name__", repr(loaded_class))
                reason = f"Class '{class_name}' does not satisfy ProtocolDomainPlugin"
                logger.warning(
                    "Plugin from entry point '%s' failed protocol check: %s",
                    ep.name,
                    reason,
                )
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep.name,
                        module_path=module_path,
                        status="protocol_invalid",
                        reason=reason,
                    )
                )
                if strict:
                    raise RuntimeError(f"Plugin from entry point '{ep.name}': {reason}")
                continue

            plugin_id = plugin.plugin_id
            if plugin_id in self._plugins:
                logger.debug(
                    "Discovered plugin '%s' from entry point '%s' "
                    "already registered (explicit wins), skipping",
                    plugin_id,
                    ep.name,
                )
                entries.append(
                    ModelPluginDiscoveryEntry(
                        entry_point_name=ep.name,
                        module_path=module_path,
                        status="duplicate_skipped",
                        plugin_id=plugin_id,
                        reason=(
                            f"Plugin ID '{plugin_id}' already registered "
                            f"(explicit registration takes precedence)"
                        ),
                    )
                )
                continue

            self._plugins[plugin_id] = plugin
            accepted.append(plugin_id)
            logger.debug(
                "Discovered and registered plugin '%s' from entry point '%s'",
                plugin_id,
                ep.name,
                extra={
                    "plugin_id": plugin_id,
                    "entry_point": ep.name,
                    "module_path": module_path,
                },
            )
            entries.append(
                ModelPluginDiscoveryEntry(
                    entry_point_name=ep.name,
                    module_path=module_path,
                    status="accepted",
                    plugin_id=plugin_id,
                )
            )

        report = ModelPluginDiscoveryReport(
            group=group,
            discovered_count=len(sorted_eps),
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

    @staticmethod
    def _validate_plugin_namespace(
        module_path: str,
        allowed_namespaces: tuple[str, ...],
    ) -> bool:
        """Validate a module path against the allowed namespace prefixes."""
        for namespace in allowed_namespaces:
            if module_path.startswith(namespace):
                if namespace.endswith("."):
                    return True
                remaining = module_path[len(namespace):]
                if remaining == "" or remaining.startswith("."):
                    return True
        return False

    @staticmethod
    def _parse_module_path(entry_point_value: str) -> str:
        """Extract the module path from an entry-point value string."""
        if ":" in entry_point_value:
            return entry_point_value.split(":", 1)[0]
        return entry_point_value


__all__: list[str] = [
    "ModelDomainPluginConfig",
    "ModelDomainPluginResult",
    "ModelHandshakeResult",
    "ProtocolDomainPlugin",
    "RegistryDomainPlugin",
]
