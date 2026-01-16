# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Compute Registry - SINGLE SOURCE OF TRUTH for compute plugin registration.

This module provides the RegistryCompute class for registering and resolving
deterministic compute plugins in the ONEX infrastructure layer.

Compute plugins:
- Perform deterministic, in-process computation with NO external I/O
- Examples: JSON normalization, ranking/scoring, policy evaluation, diffing, AST transforms
- MUST be synchronous by default (async requires explicit deterministic_async=True flag)

Design Principles:
- Single source of truth: All compute plugin registrations go through this registry
- Sync enforcement: Async plugins must be explicitly flagged
- Type-safe: Full typing for plugin registrations (no Any types)
- Thread-safe: Registration operations protected by lock
- Testable: Easy to mock and test plugin configurations

CRITICAL: Compute plugins are PURE computation logic only.

Compute plugins MUST NOT:
    - Perform I/O operations (file, network, database)
    - Have side effects (state mutation outside return values)
    - Make external service calls
    - Log at runtime
    - Depend on mutable global state
    - Access current time (unless explicitly provided as input)
    - Use random number generation (unless deterministic with provided seed)

Example Usage:
    ```python
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.registry_compute import RegistryCompute
    from omnibase_infra.runtime.models import ModelComputeRegistration

    # Container-based DI (preferred)
    container = ModelONEXContainer()
    await wire_infrastructure_services(container)
    registry = await container.service_registry.resolve_service(RegistryCompute)

    # Register a synchronous plugin using the model (preferred)
    registration = ModelComputeRegistration(
        plugin_id="json_normalizer",
        plugin_class=JsonNormalizerPlugin,
        version="1.0.0",
        description="Normalizes JSON for deterministic comparison",
    )
    registry.register(registration)

    # Register using convenience method
    registry.register_plugin(
        plugin_id="async_transformer",
        plugin_class=AsyncTransformerPlugin,
        version="1.0.0",
        deterministic_async=True,  # MUST be explicit for async plugins
    )

    # Retrieve a plugin
    plugin_cls = registry.get("json_normalizer")
    plugin = plugin_cls()
    result = plugin.execute(input_data, context)

    # List all plugins
    plugins = registry.list_keys()  # [(plugin_id, version), ...]
    ```

Integration Points:
- RuntimeHostProcess uses this registry to discover and instantiate compute plugins
- Plugins are loaded based on contract definitions
- Supports hot-reload patterns for development
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import threading
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from omnibase_infra.errors import (
    ComputeRegistryError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.runtime.models import ModelComputeRegistration
from omnibase_infra.runtime.models.model_compute_key import ModelComputeKey

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolPluginCompute, ProtocolRegistryMetrics

# Module-level logger for metrics and registry operations
logger = logging.getLogger(__name__)


# =============================================================================
# Metrics Timer Utility
# =============================================================================


class _MetricsTimer:
    """Context manager for timing operations.

    Provides high-precision timing using time.perf_counter() for accurate
    latency measurements in the registry metrics system.

    Attributes:
        elapsed_ms: Time elapsed in milliseconds after exiting the context.

    Example:
        >>> timer = _MetricsTimer()
        >>> with timer:
        ...     # perform operation
        ...     result = expensive_operation()
        >>> print(f"Operation took {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize timer with zero elapsed time."""
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> _MetricsTimer:
        """Start timing on context entry."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        """Calculate elapsed time on context exit."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# =============================================================================
# Compute Registry
# =============================================================================

# Semver sorting sentinel value (chr(127) = DEL character, ASCII 127)
# WHY chr(127): In semantic versioning, releases sort AFTER prereleases:
#   - "1.0.0-alpha" < "1.0.0" (prerelease comes before release)
#   - "1.0.0-beta" < "1.0.0"
#
# For string comparison to work correctly:
#   - Prerelease strings (e.g., "alpha", "beta") are compared lexicographically
#   - Empty prerelease (release version) needs to sort AFTER any prerelease string
#   - chr(127) is the highest printable ASCII value (DEL character)
#   - Any prerelease string ("alpha", "rc", "beta") < chr(127)
#   - Therefore: ("1.0.0-alpha") < ("1.0.0") because "alpha" < chr(127)
_SEMVER_SORT_SENTINEL = chr(127)

# Environment variable for configuring the semver LRU cache size
# Set ONEX_COMPUTE_REGISTRY_CACHE_SIZE to tune cache size for large deployments
ENV_COMPUTE_REGISTRY_CACHE_SIZE = "ONEX_COMPUTE_REGISTRY_CACHE_SIZE"

# Default cache size when environment variable is not set
_DEFAULT_SEMVER_CACHE_SIZE = 128


def _get_compute_registry_cache_size() -> int:
    """Get compute registry cache size from environment.

    Reads the ONEX_COMPUTE_REGISTRY_CACHE_SIZE environment variable.
    If not set, returns the default cache size.

    Range validation: Cache size must be between 1 and 10000.
    - Below 1: Uses default (logged as warning)
    - Above 10000: Uses default (logged as warning)

    Returns:
        Cache size as an integer.

    Raises:
        ProtocolConfigurationError: If the environment variable contains
            a non-integer value.
    """
    from omnibase_infra.enums import EnumInfraTransportType
    from omnibase_infra.utils.util_env_parsing import parse_env_int

    if os.environ.get(ENV_COMPUTE_REGISTRY_CACHE_SIZE) is not None:
        return parse_env_int(
            ENV_COMPUTE_REGISTRY_CACHE_SIZE,
            _DEFAULT_SEMVER_CACHE_SIZE,
            min_value=1,
            max_value=10000,
            transport_type=EnumInfraTransportType.HTTP,
            service_name="compute_registry",
        )

    return _DEFAULT_SEMVER_CACHE_SIZE


class RegistryCompute:
    """SINGLE SOURCE OF TRUTH for compute plugin registration in omnibase_infra.

    Thread-safe registry for compute plugins. Manages deterministic computation plugins
    that perform pure data transformations without side effects.

    The registry maintains a mapping from ModelComputeKey instances to plugin classes
    that implement the ProtocolPluginCompute protocol. ModelComputeKey provides strong
    typing and replaces the legacy tuple[str, str] pattern.

    Container Integration:
        RegistryCompute is designed to be managed by ModelONEXContainer from omnibase_core.
        Use container_wiring.wire_infrastructure_services() to register RegistryCompute
        in the container, then resolve it via:

        ```python
        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.runtime.registry_compute import RegistryCompute

        # Resolve from container (preferred)
        registry = await container.service_registry.resolve_service(RegistryCompute)
        ```

    Thread Safety:
        All registration operations are protected by a threading.Lock to ensure
        thread-safe access in concurrent environments.

    Sync Enforcement:
        By default, plugins must be synchronous. If a plugin has async methods
        (execute or any public method), registration will fail unless
        deterministic_async=True is explicitly specified.

    Scale and Performance Characteristics:

        Expected Registry Scale:
            - Typical ONEX system: 10-30 unique compute plugins across 2-5 versions each
            - Medium deployment: 30-50 plugins across 3-8 versions each
            - Large deployment: 50-100 plugins across 5-10 versions each
            - Stress tested: 500+ total registrations (100 plugins x 5 versions)

        Performance Characteristics:

            Primary Operations:
                - register(): O(1) - Direct dictionary insert with secondary index update
                - get(plugin_id): O(1) best case, O(k) average
                    where k = number of matching versions
                    - Uses secondary index (_plugin_id_index) for O(1) plugin_id lookup
                    - Fast path (single version): O(1) direct lookup
                    - Multi-version: O(k) to find max version via comparison
                    - Cached semver parsing: LRU cache (128 entries) avoids re-parsing

                - is_registered(): O(k) where k = versions for plugin_id
                - list_keys(): O(n*log n) where n = total registrations (full scan + sort)
                - list_versions(): O(k) where k = versions for plugin_id
                - unregister(): O(k) where k = versions for plugin_id

            Benchmark Targets (match RegistryPolicy):
                - 1000 sequential get() calls: < 100ms (< 0.1ms per lookup)
                - 1000 concurrent get() calls (10 threads): < 500ms
                - 100 failed lookups (missing plugin_id): < 500ms (early exit optimization)

            Lock Contention:
                - Read operations (get, is_registered): Hold lock during lookup only
                - Write operations (register, unregister): Hold lock for full operation
                - Critical sections minimized to reduce contention
                - Expected concurrent throughput: > 2000 reads/sec under 10-thread load

        Memory Footprint:

            Per Plugin Registration:
                - ModelComputeKey: ~160 bytes (2 strings: plugin_id, version)
                - Plugin class reference: 8 bytes (Python object pointer)
                - Secondary index entry: ~50 bytes (list entry + key reference)
                - Total per registration: ~220 bytes

            Estimated Registry Memory:
                - 50 registrations: ~11 KB
                - 100 registrations: ~22 KB
                - 500 registrations: ~110 KB

            Cache Overhead:
                - Semver LRU cache: Configurable via ONEX_COMPUTE_REGISTRY_CACHE_SIZE env var
                  (default: 128 entries x ~100 bytes = ~12.8 KB)
                - Total with cache: Registry memory + cache overhead

    Environment Variables:
        ONEX_COMPUTE_REGISTRY_CACHE_SIZE: Configure the semver LRU cache size for large
            deployments. Set to a higher value (e.g., 256, 512) if you have many
            unique version strings. Default: 128.

    Attributes:
        _registry: Internal dictionary mapping ModelComputeKey instances to plugin classes
        _lock: Threading lock for thread-safe registration operations
        _plugin_id_index: Secondary index for O(1) plugin_id lookup
        SEMVER_CACHE_SIZE: Class variable for LRU cache size, read from
            ONEX_COMPUTE_REGISTRY_CACHE_SIZE environment variable (default: 128)

    Example:
        >>> from omnibase_infra.runtime.models import ModelComputeRegistration
        >>> from omnibase_infra.runtime.models.model_compute_key import ModelComputeKey
        >>> registry = RegistryCompute()
        >>> registration = ModelComputeRegistration(
        ...     plugin_id="json_normalizer",
        ...     plugin_class=JsonNormalizerPlugin,
        ... )
        >>> registry.register(registration)
        >>> plugin_cls = registry.get("json_normalizer")
        >>> print(registry.list_keys())
        [('json_normalizer', '1.0.0')]
    """

    # ==========================================================================
    # Class-level semver cache configuration
    # ==========================================================================

    # Semver cache size - configurable via ONEX_COMPUTE_REGISTRY_CACHE_SIZE env var
    # Read at class definition time; can be overridden via class attribute before first parse
    SEMVER_CACHE_SIZE: int = _get_compute_registry_cache_size()

    # Cached semver parser function (lazily initialized)
    _semver_cache: Callable[[str], tuple[int, int, int, str]] | None = None

    # Lock for thread-safe cache initialization
    _semver_cache_lock: threading.Lock = threading.Lock()

    def __init__(
        self, metrics_collector: ProtocolRegistryMetrics | None = None
    ) -> None:
        """Initialize an empty compute registry with thread lock.

        Args:
            metrics_collector: Optional metrics collector for production monitoring.
                If provided, registry operations will record latency, cache hits/misses,
                registry size changes, and errors to the collector.
                See ProtocolRegistryMetrics for the expected interface.
        """
        # Key: ModelComputeKey -> plugin_class (strong typing replaces tuple pattern)
        self._registry: dict[ModelComputeKey, type[ProtocolPluginCompute]] = {}
        self._lock: threading.Lock = threading.Lock()

        # Performance optimization: Secondary indexes for O(1) lookups
        # Maps plugin_id -> list of ModelComputeKey instances
        self._plugin_id_index: dict[str, list[ModelComputeKey]] = {}

        # Optional metrics collector for production monitoring
        self._metrics: ProtocolRegistryMetrics | None = metrics_collector

    def set_metrics_collector(self, collector: ProtocolRegistryMetrics | None) -> None:
        """Set the metrics collector for production monitoring.

        This method allows setting or changing the metrics collector after
        initialization. Pass None to disable metrics collection.

        Args:
            collector: Metrics collector implementing ProtocolRegistryMetrics,
                or None to disable metrics collection.

        Example:
            >>> from omnibase_infra.runtime.registry_compute import RegistryCompute
            >>> registry = RegistryCompute()
            >>> # Later, enable metrics
            >>> registry.set_metrics_collector(my_metrics)
            >>> # Or disable
            >>> registry.set_metrics_collector(None)
        """
        self._metrics = collector

    def _validate_sync_enforcement(
        self,
        plugin_id: str,
        plugin_class: type,
        deterministic_async: bool,
    ) -> None:
        """Validate that plugin is synchronous unless explicitly flagged.

        CRITICAL: This validation inspects ALL public methods, not just execute().
        This ensures that compute plugins with any async methods are flagged properly.

        This validation enforces the synchronous-by-default plugin execution model.
        Compute plugins are expected to be pure computation logic without I/O or async
        operations. If a plugin needs async methods (e.g., for deterministic async
        computation), it must be explicitly flagged with deterministic_async=True
        during registration.

        Validation Process:
            1. Check if execute() method exists and is async
            2. Iterate through ALL public methods (not prefixed with _)
            3. Check if any method is a coroutine function
            4. If async methods found and deterministic_async=False, raise error
            5. If async methods found and deterministic_async=True, allow registration

        Args:
            plugin_id: Unique identifier for the plugin being validated
            plugin_class: The plugin class to validate for async methods
            deterministic_async: If True, allows async interface; if False, enforces sync

        Raises:
            ComputeRegistryError: If plugin has async methods and deterministic_async=False.
                                Error includes the plugin_id and the name of the async
                                method that caused validation failure.

        Example:
            >>> # This will fail - async plugin without explicit flag
            >>> class AsyncPlugin:
            ...     async def execute(self, input_data, context):
            ...         return {"result": True}
            >>> registry._validate_sync_enforcement("async_plugin", AsyncPlugin, False)
            ComputeRegistryError: Plugin 'async_plugin' has async execute() but
                                 deterministic_async=True not specified.

            >>> # This will succeed - async explicitly flagged
            >>> registry._validate_sync_enforcement("async_plugin", AsyncPlugin, True)
        """
        # First check execute() method specifically for clear error message
        if hasattr(plugin_class, "execute"):
            if asyncio.iscoroutinefunction(plugin_class.execute):
                if not deterministic_async:
                    raise ComputeRegistryError(
                        f"Plugin {plugin_id!r} has async execute() but "
                        f"deterministic_async=True not specified. "
                        f"Compute plugins must be synchronous by default.",
                        plugin_id=plugin_id,
                        context=ModelInfraErrorContext.with_correlation(
                            operation="validate_sync_enforcement",
                        ),
                        async_method="execute",
                    )

        # Check ALL public methods for async (comprehensive validation)
        for name, method in inspect.getmembers(
            plugin_class, predicate=inspect.isfunction
        ):
            # Skip private methods (prefixed with _)
            if name.startswith("_"):
                continue

            # Check if method is async
            if asyncio.iscoroutinefunction(method):
                if not deterministic_async:
                    raise ComputeRegistryError(
                        f"Plugin {plugin_id!r} has async method {name}() but "
                        f"deterministic_async=True not specified. "
                        f"Compute plugins must be synchronous by default.",
                        plugin_id=plugin_id,
                        context=ModelInfraErrorContext.with_correlation(
                            operation="validate_sync_enforcement",
                        ),
                        async_method=name,
                    )

    def register(self, registration: ModelComputeRegistration) -> None:
        """Register a compute plugin using a registration model.

        Associates a (plugin_id, version) tuple with a plugin class.
        If the combination is already registered, the existing registration is
        overwritten.

        Args:
            registration: ModelComputeRegistration containing all registration parameters:
                - plugin_id: Unique identifier for the plugin
                - plugin_class: The plugin class to register (must implement ProtocolPluginCompute)
                - version: Semantic version string (default: "1.0.0")
                - description: Human-readable description
                - deterministic_async: If True, allows async interface

        Raises:
            ComputeRegistryError: If plugin has async methods and
                               deterministic_async=False, or if plugin_class
                               does not implement ProtocolPluginCompute
            ProtocolConfigurationError: If version format is invalid

        Example:
            >>> from omnibase_infra.runtime.models import ModelComputeRegistration
            >>> registry = RegistryCompute()
            >>> registration = ModelComputeRegistration(
            ...     plugin_id="json_normalizer",
            ...     plugin_class=JsonNormalizerPlugin,
            ...     version="1.0.0",
            ... )
            >>> registry.register(registration)
        """
        # Extract fields from model
        plugin_id = registration.plugin_id
        plugin_class = registration.plugin_class
        version = registration.version
        deterministic_async = registration.deterministic_async

        # Runtime type validation: Ensure plugin_class implements ProtocolPluginCompute protocol
        # Check if execute() method exists and is callable
        execute_attr = getattr(plugin_class, "execute", None)

        if execute_attr is None:
            raise ComputeRegistryError(
                f"Plugin class {plugin_class.__name__!r} does not implement "
                f"ProtocolPluginCompute protocol: missing 'execute()' method",
                plugin_id=plugin_id,
                context=ModelInfraErrorContext.with_correlation(
                    operation="register",
                ),
                plugin_class=plugin_class.__name__,
            )

        if not callable(execute_attr):
            raise ComputeRegistryError(
                f"Plugin class {plugin_class.__name__!r} does not implement "
                f"ProtocolPluginCompute protocol: 'execute' attribute is not callable",
                plugin_id=plugin_id,
                context=ModelInfraErrorContext.with_correlation(
                    operation="register",
                ),
                plugin_class=plugin_class.__name__,
            )

        # Validate sync enforcement
        self._validate_sync_enforcement(plugin_id, plugin_class, deterministic_async)

        # Validate version format (ensures semantic versioning compliance)
        # This calls _parse_semver which will raise ProtocolConfigurationError if invalid
        self._parse_semver(version)

        # Register the plugin using ModelComputeKey
        key = ModelComputeKey(plugin_id=plugin_id, version=version)
        with self._lock:
            self._registry[key] = plugin_class
            # Update secondary index for performance optimization
            if plugin_id not in self._plugin_id_index:
                self._plugin_id_index[plugin_id] = []
            if key not in self._plugin_id_index[plugin_id]:
                self._plugin_id_index[plugin_id].append(key)
            registry_size = len(self._registry)

        # Record registry size if metrics collector is set
        if self._metrics is not None:
            try:
                self._metrics.record_registry_size(registry_size)
            except Exception:
                # Metrics recording should never break registry operations
                # WARNING level for development visibility (change to DEBUG for production)
                logger.warning(
                    "Metrics error suppressed during register()",
                    exc_info=True,
                    extra={"plugin_id": plugin_id, "version": version},
                )

    def register_plugin(
        self,
        plugin_id: str,
        plugin_class: type,
        version: str = "1.0.0",
        deterministic_async: bool = False,
        description: str = "",
    ) -> None:
        """Convenience method to register a plugin with individual parameters.

        Wraps parameters in ModelComputeRegistration and calls register().

        Args:
            plugin_id: Unique identifier for the plugin (e.g., 'json_normalizer')
            plugin_class: The plugin class to register. Must implement ProtocolPluginCompute.
            version: Semantic version string (default: "1.0.0")
            deterministic_async: If True, allows async interface. MUST be explicitly
                                flagged for plugins with async methods.
            description: Human-readable description of the plugin

        Raises:
            ComputeRegistryError: If plugin has async methods and
                               deterministic_async=False
            ProtocolConfigurationError: If version format is invalid

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin(
            ...     plugin_id="json_normalizer",
            ...     plugin_class=JsonNormalizerPlugin,
            ...     version="1.0.0",
            ... )
        """
        registration = ModelComputeRegistration(
            plugin_id=plugin_id,
            plugin_class=plugin_class,
            version=version,
            description=description,
            deterministic_async=deterministic_async,
        )
        self.register(registration)

    def get(
        self,
        plugin_id: str,
        version: str | None = None,
    ) -> type[ProtocolPluginCompute]:
        """Get compute plugin by ID and optional version.

        Resolves the plugin class registered for the given plugin configuration.
        If version is not specified, returns the latest version (semver sorted).

        Performance Characteristics:
            - Best case: O(1) - Direct lookup with single version
            - Average case: O(k) where k = number of matching versions
            - Uses secondary index for O(1) plugin_id lookup instead of O(n) scan
            - Defers expensive error message generation until actually needed

        Args:
            plugin_id: Plugin identifier.
            version: Optional version filter. If None, returns latest version.

        Returns:
            Plugin class registered for the configuration.

        Raises:
            ComputeRegistryError: If no matching plugin is found.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerPlugin)
            >>> plugin_cls = registry.get("normalizer")
            >>> plugin_cls = registry.get("normalizer", version="1.0.0")
        """
        timer = _MetricsTimer()
        try:
            with timer:
                with self._lock:
                    # Performance optimization: Use secondary index for O(1) lookup
                    # This avoids iterating through all registry entries (O(n) -> O(1))
                    candidate_keys = self._plugin_id_index.get(plugin_id, [])

                    # Early exit if plugin_id not found - avoid building matches list
                    if not candidate_keys:
                        # Record error before raising
                        if self._metrics is not None:
                            try:
                                self._metrics.record_error("not_found", plugin_id)
                            except Exception:
                                # Metrics recording should never break registry operations
                                # WARNING level for development visibility (change to DEBUG for production)
                                logger.warning(
                                    "Metrics error suppressed during get() not_found",
                                    exc_info=True,
                                    extra={"plugin_id": plugin_id},
                                )
                        # Get unique, sorted list of registered plugin IDs for error context
                        # Uses secondary index for O(m) where m=unique plugins vs O(n) scan
                        registered: list[str] = sorted(self._plugin_id_index.keys())
                        raise ComputeRegistryError(
                            f"No compute plugin registered with id={plugin_id!r}. "
                            f"Registered plugins: {registered}",
                            plugin_id=plugin_id,
                            registered_plugins=registered,
                            context=ModelInfraErrorContext.with_correlation(
                                operation="get",
                            ),
                            version=version,
                        )

                    # If version specified, do exact match
                    if version is not None:
                        for key in candidate_keys:
                            if key.version == version:
                                return self._registry[key]
                        # Record error before raising
                        if self._metrics is not None:
                            try:
                                self._metrics.record_error(
                                    "version_not_found", plugin_id
                                )
                            except Exception:
                                # Metrics recording should never break registry operations
                                # WARNING level for development visibility (change to DEBUG for production)
                                logger.warning(
                                    "Metrics error suppressed during get() version_not_found",
                                    exc_info=True,
                                    extra={"plugin_id": plugin_id, "version": version},
                                )
                        # Version not found - get versions from candidate_keys
                        available_versions = sorted(
                            [key.version for key in candidate_keys],
                            key=self._parse_semver,
                        )
                        raise ComputeRegistryError(
                            f"Compute plugin {plugin_id!r} version {version!r} "
                            f"not found. Available versions: {available_versions}",
                            plugin_id=plugin_id,
                            context=ModelInfraErrorContext.with_correlation(
                                operation="get",
                            ),
                            version=version,
                        )

                    # Return latest version (no version filter)
                    # Fast path optimization: avoid sorting if only one version
                    if len(candidate_keys) == 1:
                        return self._registry[candidate_keys[0]]

                    # Multiple versions - find latest using semver comparison
                    latest_key = max(
                        candidate_keys,
                        key=lambda k: self._parse_semver(k.version),
                    )
                    return self._registry[latest_key]
        finally:
            # Record latency if metrics collector is set
            if self._metrics is not None:
                try:
                    self._metrics.record_get_latency(
                        plugin_id, version, timer.elapsed_ms
                    )
                except Exception:
                    # Metrics recording should never break registry operations
                    # WARNING level for development visibility (change to DEBUG for production)
                    logger.warning(
                        "Metrics error suppressed during get() latency recording",
                        exc_info=True,
                        extra={"plugin_id": plugin_id, "version": version},
                    )

    def list_keys(self) -> list[tuple[str, str]]:
        """List registered plugin keys as (plugin_id, version) tuples.

        Returns:
            List of (plugin_id, version) tuples, sorted by plugin_id then semver.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerV1, version="1.0.0")
            >>> registry.register_plugin("normalizer", NormalizerV2, version="2.0.0")
            >>> print(registry.list_keys())
            [('normalizer', '1.0.0'), ('normalizer', '2.0.0')]
        """
        with self._lock:
            return sorted(
                [k.to_tuple() for k in self._registry],
                key=lambda x: (x[0], self._parse_semver(x[1])),
            )

    def list_versions(self, plugin_id: str) -> list[str]:
        """List registered versions for a plugin ID.

        Args:
            plugin_id: The plugin ID to list versions for.

        Returns:
            List of version strings registered for the plugin ID, sorted by semver.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerV1, version="1.0.0")
            >>> registry.register_plugin("normalizer", NormalizerV2, version="2.0.0")
            >>> print(registry.list_versions("normalizer"))
            ['1.0.0', '2.0.0']
        """
        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._plugin_id_index.get(plugin_id, [])
            versions = sorted(
                [key.version for key in candidate_keys],
                key=self._parse_semver,
            )
            return versions

    def is_registered(
        self,
        plugin_id: str,
        version: str | None = None,
    ) -> bool:
        """Check if a plugin is registered.

        Args:
            plugin_id: Plugin identifier.
            version: Optional version filter.

        Returns:
            True if a matching plugin is registered, False otherwise.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerPlugin)
            >>> registry.is_registered("normalizer")
            True
            >>> registry.is_registered("unknown")
            False
        """
        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._plugin_id_index.get(plugin_id, [])
            if not candidate_keys:
                return False
            if version is None:
                return True
            return any(key.version == version for key in candidate_keys)

    def unregister(
        self,
        plugin_id: str,
        version: str | None = None,
    ) -> int:
        """Unregister compute plugin(s).

        Removes plugin registrations matching the given criteria.
        This is useful for testing and hot-reload scenarios.

        Args:
            plugin_id: Plugin identifier to unregister.
            version: Optional version filter. If None, removes all versions.

        Returns:
            Number of plugins unregistered.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerV1, version="1.0.0")
            >>> registry.register_plugin("normalizer", NormalizerV2, version="2.0.0")
            >>> registry.unregister("normalizer")  # Removes all versions
            2
            >>> registry.unregister("normalizer", version="1.0.0")  # Remove specific version
            1
        """
        # Thread safety: Lock held during full unregister operation (write operation)
        with self._lock:
            # Performance optimization: Use secondary index
            candidate_keys = self._plugin_id_index.get(plugin_id, [])
            keys_to_remove: list[ModelComputeKey] = []

            for key in candidate_keys:
                if version is None or key.version == version:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._registry[key]
                # Update secondary index
                self._plugin_id_index[plugin_id].remove(key)

            # Clean up empty index entries
            if (
                plugin_id in self._plugin_id_index
                and not self._plugin_id_index[plugin_id]
            ):
                del self._plugin_id_index[plugin_id]

            registry_size = len(self._registry)
            removed_count = len(keys_to_remove)

        # Record registry size if metrics collector is set and we removed something
        if removed_count > 0 and self._metrics is not None:
            try:
                self._metrics.record_registry_size(registry_size)
            except Exception:
                # Metrics recording should never break registry operations
                # WARNING level for development visibility (change to DEBUG for production)
                logger.warning(
                    "Metrics error suppressed during unregister()",
                    exc_info=True,
                    extra={"plugin_id": plugin_id, "version": version},
                )

        return removed_count

    def clear(self) -> None:
        """Clear all plugin registrations.

        Removes all registered plugins from the registry.

        Warning:
            This method is intended for **testing purposes only**.
            Calling it in production code will emit a warning.
            It breaks the immutability guarantee after startup.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerPlugin)
            >>> registry.clear()
            >>> registry.list_keys()
            []
        """
        warnings.warn(
            "RegistryCompute.clear() is intended for testing only. "
            "Do not use in production code.",
            UserWarning,
            stacklevel=2,
        )
        with self._lock:
            self._registry.clear()
            self._plugin_id_index.clear()

    def __len__(self) -> int:
        """Return the number of registered plugins.

        Returns:
            Number of registered plugin (plugin_id, version) combinations.

        Example:
            >>> registry = RegistryCompute()
            >>> len(registry)
            0
            >>> registry.register_plugin("normalizer", NormalizerPlugin)
            >>> len(registry)
            1
        """
        with self._lock:
            return len(self._registry)

    def __contains__(self, key: ModelComputeKey | str) -> bool:
        """Check if plugin is registered using 'in' operator.

        Args:
            key: Either a ModelComputeKey instance or a plugin_id string.

        Returns:
            True if plugin is registered, False otherwise.

        Example:
            >>> registry = RegistryCompute()
            >>> registry.register_plugin("normalizer", NormalizerPlugin)
            >>> "normalizer" in registry
            True
            >>> ModelComputeKey(plugin_id="normalizer", version="1.0.0") in registry
            True
            >>> "unknown" in registry
            False
        """
        if isinstance(key, str):
            return self.is_registered(key)
        return self.is_registered(key.plugin_id, key.version)

    # ==========================================================================
    # Semver Cache Configuration Methods
    # ==========================================================================

    @classmethod
    def _get_semver_parser(cls) -> Callable[[str], tuple[int, int, int, str]]:
        """Get or create the semver parser with configured cache size.

        This method implements lazy initialization of the LRU-cached semver parser.
        The cache size is determined by SEMVER_CACHE_SIZE at initialization time.

        Thread Safety:
            Uses double-checked locking pattern for thread-safe lazy initialization.
            The fast path stores the cache reference in a local variable to prevent
            TOCTOU (time-of-check-time-of-use) race conditions where another thread
            could call _reset_semver_cache() between the None check and the return.

        Returns:
            Cached semver parsing function.

        Performance:
            - First call: Creates LRU-cached function (one-time cost)
            - Subsequent calls: Returns cached function reference (O(1))
        """
        # Fast path: cache already initialized
        # CRITICAL: Store in local variable to prevent TOCTOU race condition.
        # Without this, another thread could call _reset_semver_cache() between
        # the None check and the return, causing this method to return None.
        cache = cls._semver_cache
        if cache is not None:
            return cache

        # Slow path: initialize with lock
        with cls._semver_cache_lock:
            # Double-check after acquiring lock
            if cls._semver_cache is not None:
                return cls._semver_cache

            # Create LRU-cached parser with configured size
            @functools.lru_cache(maxsize=cls.SEMVER_CACHE_SIZE)
            def _parse_semver_impl(version: str) -> tuple[int, int, int, str]:
                """Parse semantic version string into comparable tuple.

                Implementation moved here to support configurable cache size.
                See _parse_semver docstring for full documentation.
                """
                # Validate non-empty version string
                if not version or not version.strip():
                    raise ProtocolConfigurationError(
                        "Invalid semantic version format: empty version string",
                        context=ModelInfraErrorContext.with_correlation(
                            operation="parse_semver",
                        ),
                        version=version,
                    )

                # Trim whitespace BEFORE any split operations
                version = version.strip()

                # Split off prerelease suffix (e.g., "1.0.0-alpha" -> "1.0.0", "alpha")
                if "-" in version:
                    version_part, prerelease = version.split("-", 1)
                    # Validate prerelease is non-empty when dash is present
                    if not prerelease:
                        raise ProtocolConfigurationError(
                            f"Invalid semantic version format: '{version}'. "
                            f"Prerelease suffix cannot be empty when '-' is specified.",
                            context=ModelInfraErrorContext.with_correlation(
                                operation="parse_semver",
                            ),
                            version=version,
                        )
                else:
                    version_part, prerelease = version, ""

                # Parse major.minor.patch
                parts = version_part.split(".")

                # Validate version format (max 3 parts, no empty parts)
                # Note: len(parts) >= 1 is guaranteed since split(".") always returns
                # at least one element, so we only need to check the upper bound
                if len(parts) > 3 or any(not p.strip() for p in parts):
                    raise ProtocolConfigurationError(
                        f"Invalid semantic version format: '{version}'",
                        context=ModelInfraErrorContext.with_correlation(
                            operation="parse_semver",
                        ),
                        version=version,
                    )

                try:
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0
                    patch = int(parts[2]) if len(parts) > 2 else 0
                except (ValueError, IndexError) as e:
                    raise ProtocolConfigurationError(
                        f"Invalid semantic version format: '{version}'",
                        context=ModelInfraErrorContext.with_correlation(
                            operation="parse_semver",
                        ),
                        version=version,
                    ) from e

                # Validate non-negative integers
                if major < 0 or minor < 0 or patch < 0:
                    raise ProtocolConfigurationError(
                        f"Invalid semantic version: negative component in '{version}'",
                        context=ModelInfraErrorContext.with_correlation(
                            operation="parse_semver",
                        ),
                        version=version,
                    )

                # Empty prerelease uses sentinel (chr(127)) to sort AFTER any prerelease string
                # This ensures "1.0.0" > "1.0.0-alpha" in version comparisons
                sort_prerelease = prerelease if prerelease else _SEMVER_SORT_SENTINEL

                return (major, minor, patch, sort_prerelease)

            cls._semver_cache = _parse_semver_impl
            return cls._semver_cache

    @classmethod
    def _parse_semver(cls, version: str) -> tuple[int, int, int, str]:
        """Parse semantic version string into comparable tuple with INTEGER components.

        This method implements SEMANTIC VERSION SORTING, not lexicographic sorting.
        This is critical for correct "latest version" selection.

        Why This Matters:
            Lexicographic sorting (string comparison):
                "1.10.0" < "1.9.0" WRONG (because '1' < '9' in strings)
                "10.0.0" < "2.0.0" WRONG (because '1' < '2' in strings)

            Semantic version sorting (integer comparison):
                1.10.0 > 1.9.0 CORRECT (because 10 > 9 as integers)
                10.0.0 > 2.0.0 CORRECT (because 10 > 2 as integers)

        Implementation:
            - Parses version components as INTEGERS (not strings)
            - Returns tuple (major: int, minor: int, patch: int, prerelease: str)
            - Python's tuple comparison then works correctly: (1, 10, 0) > (1, 9, 0)
            - Prerelease versions sort before release: "1.0.0-alpha" < "1.0.0"

        Supported Formats:
            - Full: "1.2.3", "1.2.3-beta"
            - Partial: "1" -> (1, 0, 0), "1.2" -> (1, 2, 0)
            - Prerelease: "1.0.0-alpha", "2.1.0-rc.1"

        Performance:
            This method uses an LRU cache with configurable size (default: 128)
            to avoid re-parsing the same version strings repeatedly.

        Args:
            version: Semantic version string (e.g., "1.2.3" or "1.0.0-beta")

        Returns:
            Tuple of (major, minor, patch, prerelease) for comparison.
            Components are INTEGERS (not strings) for correct semantic sorting.

        Raises:
            ProtocolConfigurationError: If version format is invalid
        """
        parser = cls._get_semver_parser()
        return parser(version)

    @classmethod
    def _reset_semver_cache(cls) -> None:
        """Reset semver cache. For testing only.

        Clears the cached semver parser, allowing reconfiguration of cache size.
        This should only be used in test fixtures to ensure test isolation.

        Thread Safety:
            This method is thread-safe and uses the class-level lock. The reset
            operation is atomic - either the cache is fully reset or not at all.

            In-flight Operations:
                If other threads have already obtained a reference to the cache
                via _get_semver_parser(), they will continue using the old cache
                until they complete. This is safe because the old cache remains
                a valid callable until garbage collected. New operations after
                reset will get the new cache instance when created.

            Memory Reclamation:
                The old cache's internal LRU entries are explicitly cleared via
                cache_clear() before the reference is released. This ensures
                prompt memory reclamation rather than waiting for garbage
                collection.

            Concurrent Reset:
                Multiple concurrent reset calls are safe. Each reset will clear
                the current cache (if any) and set the reference to None. The
                lock ensures only one reset executes at a time.

        Example:
            >>> # In test fixture
            >>> RegistryCompute._reset_semver_cache()
            >>> RegistryCompute.SEMVER_CACHE_SIZE = 64
            >>> # Now cache will be initialized with size 64 on next use
        """
        with cls._semver_cache_lock:
            old_cache = cls._semver_cache
            if old_cache is not None:
                # Clear internal LRU cache entries before releasing reference.
                # This ensures prompt memory reclamation rather than waiting
                # for garbage collection of the orphaned function object.
                # NOTE: cache_clear() is added by @lru_cache decorator but not
                # reflected in Callable type annotation. This is a known mypy
                # limitation with lru_cache wrappers.
                old_cache.cache_clear()  # type: ignore[attr-defined]  # NOTE: lru_cache dynamic method
            cls._semver_cache = None


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    "ENV_COMPUTE_REGISTRY_CACHE_SIZE",  # Environment variable constant
    "ModelComputeKey",  # Re-export for convenience
    "ModelComputeRegistration",  # Re-export for convenience
    "RegistryCompute",  # Registry class
]
