# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding configuration resolver for ONEX infrastructure.

BindingConfigResolver provides a unified interface for resolving handler configurations
from multiple sources with proper priority ordering (highest to lowest):

    1. Environment variables (HANDLER_{TYPE}_{FIELD}) - **highest priority**, always wins
    2. Inline config (passed directly to resolve()) - overrides config_ref values
    3. Config reference (file:, env:, vault:) - **base configuration** (lowest priority)

Resolution Process:
    The resolver builds the final configuration by layering sources from lowest to highest
    priority. Later sources override earlier ones for overlapping keys:

    1. Load base config from config_ref (if provided)
    2. Merge inline_config on top (inline values override config_ref values)
    3. Apply environment variable overrides on top (env values override everything)

    This means: config_ref provides defaults, inline_config can override those defaults,
    and environment variables can override both for operational flexibility.

Important: config_ref Schemes are Mutually Exclusive
    The config_ref schemes (file:, env:, vault:) are **mutually exclusive** - only ONE
    config_ref can be provided per resolution call. The scheme determines WHERE to load
    the base configuration from, not a priority ordering between schemes.

    Correct usage::

        # Load from file
        resolver.resolve(handler_type="db", config_ref="file:configs/db.yaml")

        # OR load from environment variable
        resolver.resolve(handler_type="db", config_ref="env:DB_CONFIG_JSON")

        # OR load from Vault
        resolver.resolve(handler_type="db", config_ref="vault:secret/data/db")

    Invalid usage (would use only ONE, ignoring others)::

        # WRONG: Cannot combine multiple config_ref schemes in a single call
        # config_ref="file:..." AND config_ref="env:..." is NOT supported
        # Pass only ONE config_ref string per resolve() call

Design Philosophy:
    - Dumb and deterministic: resolves and caches, does not discover or mutate
    - Environment overrides always take precedence for operational flexibility
    - Caching is optional and TTL-controlled for performance vs freshness tradeoff

Example:
    Basic usage with container-based dependency injection::

        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.runtime.util_container_wiring import wire_infrastructure_services

        # Bootstrap container and register config
        container = ModelONEXContainer()
        config = ModelBindingConfigResolverConfig(env_prefix="HANDLER")
        from omnibase_core.enums import EnumInjectionScope

        await container.service_registry.register_instance(
            interface=ModelBindingConfigResolverConfig,
            instance=config,
            scope=EnumInjectionScope.GLOBAL,
        )

        # Create resolver with container injection
        resolver = BindingConfigResolver(container)

        # Resolve from inline config
        binding = resolver.resolve(
            handler_type="db",
            inline_config={"pool_size": 10, "timeout_ms": 5000}
        )

        # Resolve from file reference
        binding = resolver.resolve(
            handler_type="vault",
            config_ref="file:configs/vault.yaml"
        )

    With environment overrides::

        # Set HANDLER_DB_TIMEOUT_MS=10000 in environment
        binding = resolver.resolve(
            handler_type="db",
            inline_config={"timeout_ms": 5000}  # Will be overridden to 10000
        )

Security Considerations:
    - File paths are validated to prevent path traversal attacks
    - Error messages are sanitized to exclude configuration values
    - Vault secrets are resolved through SecretResolver (not accessed directly)
    - File size limits prevent memory exhaustion attacks

Thread Safety:
    This class supports concurrent access from both sync and async contexts
    using a two-level locking strategy:

    1. ``threading.Lock`` (``_lock``): Protects all cache reads/writes and
       stats updates. This lock is held briefly for in-memory operations.

    2. Per-key ``asyncio.Lock`` (``_async_key_locks``): Prevents duplicate
       async fetches for the SAME handler type. When multiple async callers
       request the same config simultaneously, only one performs the fetch
       while others wait and reuse the cached result.

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    SecretResolutionError,
)
from omnibase_infra.runtime.models.model_binding_config import ModelBindingConfig
from omnibase_infra.runtime.models.model_binding_config_cache_stats import (
    ModelBindingConfigCacheStats,
)
from omnibase_infra.runtime.models.model_binding_config_resolver_config import (
    ModelBindingConfigResolverConfig,
)
from omnibase_infra.runtime.models.model_config_cache_entry import ModelConfigCacheEntry
from omnibase_infra.runtime.models.model_config_ref import (
    EnumConfigRefScheme,
    ModelConfigRef,
)
from omnibase_infra.runtime.models.model_retry_policy import ModelRetryPolicy

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.runtime.secret_resolver import SecretResolver

logger = logging.getLogger(__name__)

# Maximum file size for config files (1MB)
# Prevents memory exhaustion from accidentally pointing at large files
MAX_CONFIG_FILE_SIZE: Final[int] = 1024 * 1024

# Maximum recursion depth for nested config resolution
# Prevents stack overflow on deeply nested or circular configs
_MAX_NESTED_CONFIG_DEPTH: Final[int] = 20

# Fields that can be overridden via environment variables
# Maps from environment variable field name (uppercase) to model field name
_ENV_OVERRIDE_FIELDS: Final[dict[str, str]] = {
    "ENABLED": "enabled",
    "PRIORITY": "priority",
    "TIMEOUT_MS": "timeout_ms",
    "RATE_LIMIT_PER_SECOND": "rate_limit_per_second",
    "MAX_RETRIES": "max_retries",
    "BACKOFF_STRATEGY": "backoff_strategy",
    "BASE_DELAY_MS": "base_delay_ms",
    "MAX_DELAY_MS": "max_delay_ms",
    "NAME": "name",
}

# Retry policy fields (nested under retry_policy)
_RETRY_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    {"MAX_RETRIES", "BACKOFF_STRATEGY", "BASE_DELAY_MS", "MAX_DELAY_MS"}
)

# Async key lock cleanup configuration (default values)
# These values are now configurable via ModelBindingConfigResolverConfig.
# These constants are kept as fallbacks and for backward compatibility with tests
# that directly manipulate internal state without going through config.
# Prevents unbounded growth of _async_key_locks dict in long-running processes.
_ASYNC_KEY_LOCK_CLEANUP_THRESHOLD: Final[int] = (
    1000  # Trigger cleanup when > 1000 locks (default, can be overridden via config)
)
_ASYNC_KEY_LOCK_MAX_AGE_SECONDS: Final[float] = (
    3600.0  # Clean locks older than 1 hour (default, can be overridden via config)
)


def _split_path_and_fragment(path: str) -> tuple[str, str | None]:
    """Split a path into path and optional fragment at '#'.

    This is a common helper used by both BindingConfigResolver and SecretResolver
    to parse vault paths that may contain a fragment identifier (e.g., "path#field").

    Args:
        path: The path string, optionally containing a '#' separator.

    Returns:
        Tuple of (path, fragment) where fragment may be None if no '#' present.

    Example:
        >>> _split_path_and_fragment("secret/data/db#password")
        ("secret/data/db", "password")
        >>> _split_path_and_fragment("secret/data/config")
        ("secret/data/config", None)
    """
    if "#" in path:
        path_part, fragment = path.rsplit("#", 1)
        return path_part, fragment
    return path, None


class BindingConfigResolver:  # ONEX_EXCLUDE: method_count - follows SecretResolver pattern
    """Resolver that normalizes handler configs from multiple sources.

    The BindingConfigResolver provides a unified interface for resolving handler
    configurations with proper priority ordering and caching support.

    Resolution Order:
        1. Check cache (if enabled and not expired)
        2. Parse config_ref if present (file:, env:, vault:)
        3. Load base config from ref, then merge inline_config (inline takes precedence)
        4. Apply environment variable overrides (highest priority)
        5. Resolve any vault: references in config values
        6. Validate and construct ModelBindingConfig

    Thread Safety:
        This class is thread-safe for concurrent access from both sync and
        async contexts. See module docstring for details on the locking strategy.

    Example:
        >>> # Container setup (async context required)
        >>> from omnibase_core.enums import EnumInjectionScope
        >>> container = ModelONEXContainer()
        >>> config = ModelBindingConfigResolverConfig(env_prefix="HANDLER")
        >>> await container.service_registry.register_instance(
        ...     interface=ModelBindingConfigResolverConfig,
        ...     instance=config,
        ...     scope=EnumInjectionScope.GLOBAL,
        ... )
        >>> resolver = await BindingConfigResolver.create(container)
        >>> binding = resolver.resolve(
        ...     handler_type="db",
        ...     inline_config={"pool_size": 10}
        ... )
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        *,
        _config: ModelBindingConfigResolverConfig | None = None,
        _secret_resolver: SecretResolver | None = None,
    ) -> None:
        """Initialize BindingConfigResolver with container-based dependency injection.

        Follows ONEX mandatory container injection pattern per CLAUDE.md.
        Config is resolved from container's service registry, and SecretResolver
        is resolved as an optional dependency.

        Note:
            Prefer using the async factory method ``create()`` for initialization,
            which properly resolves dependencies from the container's service registry.
            Direct ``__init__`` usage requires pre-resolved config and secret_resolver.

        Args:
            container: ONEX container for dependency resolution.
            _config: Pre-resolved config (used by create() factory). If None, raises error.
            _secret_resolver: Pre-resolved secret resolver (optional, used by create() factory).

        Raises:
            ProtocolConfigurationError: If _config is not provided (use create() instead).
        """
        self._container = container

        # Validate that config was provided (either via create() or directly)
        if _config is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="init",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "BindingConfigResolver requires config to be provided. "
                "Use the async factory method BindingConfigResolver.create(container) "
                "for proper initialization with dependency resolution.",
                context=context,
            )

        self._config: ModelBindingConfigResolverConfig = _config
        self._secret_resolver: SecretResolver | None = _secret_resolver

        # Use OrderedDict for LRU eviction support - entries are moved to end on access
        self._cache: OrderedDict[str, ModelConfigCacheEntry] = OrderedDict()
        # Track mutable stats internally since ModelBindingConfigCacheStats is frozen
        self._hits = 0
        self._misses = 0
        self._expired_evictions = 0
        self._lru_evictions = 0
        self._refreshes = 0
        self._file_loads = 0
        self._env_loads = 0
        self._vault_loads = 0
        self._async_key_lock_cleanups = 0  # Track cleanup events for observability

        # RLock (Reentrant Lock) is REQUIRED - DO NOT CHANGE TO REGULAR LOCK.
        #
        # Why RLock is necessary:
        # -----------------------
        # The sync path (resolve()) holds the lock while calling internal methods
        # that also need to update counters protected by the same lock:
        #
        #   resolve() [holds _lock]
        #     -> _get_from_cache() [updates _hits, _expired_evictions]
        #     -> _resolve_config() [no lock needed directly]
        #       -> _load_from_file() [updates _file_loads]
        #       -> _load_from_env() [updates _env_loads]
        #       -> _resolve_vault_refs() [updates _vault_loads]
        #     -> _cache_config() [updates _misses, _lru_evictions]
        #
        # With a regular threading.Lock, this would cause DEADLOCK because:
        # - Thread A calls resolve() and acquires _lock
        # - Thread A calls _get_from_cache() which tries to update _hits
        # - Since Thread A already holds _lock, a regular Lock would block forever
        #
        # RLock allows the same thread to acquire the lock multiple times,
        # with a release count that must match the acquisition count.
        #
        # Alternative considered: Move counter updates outside the critical section.
        # This was rejected because:
        # 1. Counters must be updated atomically with cache operations for consistency
        # 2. Would require significant refactoring with subtle race condition risks
        # 3. RLock performance overhead is minimal for in-memory operations
        #
        # See PR #168 review for detailed analysis of this design decision.
        self._lock = threading.RLock()

        # Per-key async locks to allow parallel fetches for different handler types
        # while preventing duplicate fetches for the same handler type.
        # Timestamps track when each lock was created for periodic cleanup.
        self._async_key_locks: dict[str, asyncio.Lock] = {}
        self._async_key_lock_timestamps: dict[str, float] = {}

    @classmethod
    async def create(
        cls,
        container: ModelONEXContainer,
    ) -> BindingConfigResolver:
        """Async factory method for creating BindingConfigResolver with proper DI.

        This is the preferred method for creating BindingConfigResolver instances.
        It properly resolves dependencies from the container's async service registry.

        Args:
            container: ONEX container for dependency resolution.

        Returns:
            Fully initialized BindingConfigResolver instance.

        Raises:
            ProtocolConfigurationError: If ModelBindingConfigResolverConfig is not registered
                in the container's service registry.

        Example:
            >>> from omnibase_core.enums import EnumInjectionScope
            >>> container = ModelONEXContainer()
            >>> config = ModelBindingConfigResolverConfig(env_prefix="HANDLER")
            >>> await container.service_registry.register_instance(
            ...     interface=ModelBindingConfigResolverConfig,
            ...     instance=config,
            ...     scope=EnumInjectionScope.GLOBAL,
            ... )
            >>> resolver = await BindingConfigResolver.create(container)
        """
        # Resolve config from container's service registry
        try:
            config: ModelBindingConfigResolverConfig = (
                await container.service_registry.resolve_service(
                    ModelBindingConfigResolverConfig
                )
            )
        except (LookupError, KeyError, TypeError, AttributeError) as e:
            # LookupError/KeyError: service not registered
            # TypeError: invalid interface specification
            # AttributeError: container/registry missing expected methods
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="create",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Failed to resolve ModelBindingConfigResolverConfig from container. "
                "Ensure config is registered via container.service_registry.register_instance().",
                context=context,
            ) from e

        # Resolve SecretResolver from container (optional dependency)
        secret_resolver: SecretResolver | None = None
        try:
            from omnibase_infra.runtime.secret_resolver import SecretResolver

            secret_resolver = await container.service_registry.resolve_service(
                SecretResolver
            )
        except (ImportError, KeyError, AttributeError):
            # SecretResolver is optional - if not registered, vault: schemes won't work
            # ImportError: SecretResolver module not available
            # KeyError: SecretResolver not registered in service registry
            # AttributeError: service_registry missing resolve_service method (test mocks)
            pass

        return cls(
            container,
            _config=config,
            _secret_resolver=secret_resolver,
        )

    # === Primary API (Sync) ===

    def resolve(
        self,
        handler_type: str,
        config_ref: str | None = None,
        inline_config: dict[str, JsonType] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelBindingConfig:
        """Resolve handler configuration synchronously.

        Resolution order:
            1. Check cache (if enabled and not expired)
            2. Load from config_ref (if provided)
            3. Merge with inline_config (inline takes precedence)
            4. Apply environment variable overrides (highest priority)
            5. Validate and construct ModelBindingConfig

        Args:
            handler_type: Handler type identifier (e.g., "db", "vault", "consul").
            config_ref: Optional reference to external configuration.
                Supported schemes: file:, env:, vault: (mutually exclusive - use only ONE)
                Examples: file:configs/db.yaml, env:DB_CONFIG, vault:secret/data/db#password
            inline_config: Optional inline configuration dictionary.
                Takes precedence over config_ref for overlapping keys.
            correlation_id: Optional correlation ID for error tracking.

        Returns:
            Resolved and validated ModelBindingConfig.

        Raises:
            ProtocolConfigurationError: If configuration is invalid or cannot be loaded.
        """
        correlation_id = correlation_id or uuid4()

        with self._lock:
            # Check cache first
            cached = self._get_from_cache(handler_type)
            if cached is not None:
                return cached

            # Resolve from sources
            result = self._resolve_config(
                handler_type=handler_type,
                config_ref=config_ref,
                inline_config=inline_config,
                correlation_id=correlation_id,
            )

            # Cache the result if caching is enabled
            if self._config.enable_caching:
                source = self._describe_source(config_ref, inline_config)
                self._cache_config(handler_type, result, source)
            else:
                # Count miss when caching is disabled since _cache_config won't be called
                self._misses += 1

            return result

    def resolve_many(
        self,
        bindings: list[dict[str, JsonType]],
        correlation_id: UUID | None = None,
    ) -> list[ModelBindingConfig]:
        """Resolve multiple handler configurations.

        Each binding dict must contain at least "handler_type" key.
        Optionally can include "config_ref" and "config" (inline_config).

        Args:
            bindings: List of binding specifications. Each dict should contain:
                - handler_type (required): Handler type identifier
                - config_ref (optional): Reference to external configuration
                - config (optional): Inline configuration dictionary
            correlation_id: Optional correlation ID for error tracking.

        Returns:
            List of resolved ModelBindingConfig instances.

        Raises:
            ProtocolConfigurationError: If any configuration is invalid.

        Note:
            This sync method resolves configurations sequentially. For better
            latency when resolving multiple configurations that involve I/O
            (file or Vault), prefer using ``resolve_many_async()``.
        """
        correlation_id = correlation_id or uuid4()
        results: list[ModelBindingConfig] = []

        for binding in bindings:
            handler_type = binding.get("handler_type")
            if not isinstance(handler_type, str):
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="resolve_many",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Each binding must have a 'handler_type' string field",
                    context=context,
                )

            config_ref = binding.get("config_ref")
            if config_ref is not None and not isinstance(config_ref, str):
                config_ref = None

            inline_config = binding.get("config")
            if inline_config is not None and not isinstance(inline_config, dict):
                inline_config = None

            result = self.resolve(
                handler_type=handler_type,
                config_ref=config_ref,
                inline_config=inline_config,
                correlation_id=correlation_id,
            )
            results.append(result)

        return results

    # === Primary API (Async) ===

    async def resolve_async(
        self,
        handler_type: str,
        config_ref: str | None = None,
        inline_config: dict[str, JsonType] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelBindingConfig:
        """Resolve handler configuration asynchronously.

        For file-based configs, this uses async file I/O. For Vault secrets,
        this uses the SecretResolver's async interface.

        Thread Safety:
            Uses threading.Lock for cache access to prevent race conditions
            with sync callers. Per-key async locks serialize resolution for the
            same handler type while allowing parallel fetches for different types.

        Args:
            handler_type: Handler type identifier (e.g., "db", "vault", "consul").
            config_ref: Optional reference to external configuration.
                Supported schemes: file:, env:, vault: (mutually exclusive - use only ONE)
            inline_config: Optional inline configuration dictionary.
            correlation_id: Optional correlation ID for error tracking.

        Returns:
            Resolved and validated ModelBindingConfig.

        Raises:
            ProtocolConfigurationError: If configuration is invalid or cannot be loaded.
        """
        correlation_id = correlation_id or uuid4()

        # Use threading lock for cache check (fast operation, prevents race with sync)
        with self._lock:
            cached = self._get_from_cache(handler_type)
            if cached is not None:
                return cached

        # Get or create per-key async lock for this handler_type
        key_lock = self._get_async_key_lock(handler_type)

        async with key_lock:
            # Double-check cache after acquiring async lock
            with self._lock:
                cached = self._get_from_cache(handler_type)
                if cached is not None:
                    return cached

            # Resolve from sources asynchronously
            result = await self._resolve_config_async(
                handler_type=handler_type,
                config_ref=config_ref,
                inline_config=inline_config,  # type: ignore[arg-type]
                correlation_id=correlation_id,
            )

            # Cache the result if caching is enabled
            if self._config.enable_caching:
                with self._lock:
                    if handler_type not in self._cache:
                        source = self._describe_source(config_ref, inline_config)
                        self._cache_config(handler_type, result, source)
            else:
                # Count miss when caching is disabled since _cache_config won't be called
                with self._lock:
                    self._misses += 1

            return result

    async def resolve_many_async(
        self,
        bindings: list[dict[str, JsonType]],
        correlation_id: UUID | None = None,
    ) -> list[ModelBindingConfig]:
        """Resolve multiple configurations asynchronously in parallel.

        Uses asyncio.gather() to fetch multiple configurations concurrently,
        improving performance when resolving multiple configs that may involve
        I/O (e.g., file or Vault-based secrets).

        Thread Safety:
            Each configuration resolution uses per-key async locks, so fetches
            for different handler types proceed in parallel while fetches for
            the same handler type are serialized.

        Args:
            bindings: List of binding specifications.
            correlation_id: Optional correlation ID for error tracking.

        Returns:
            List of resolved ModelBindingConfig instances.

        Raises:
            ProtocolConfigurationError: If any configuration is invalid.
        """
        correlation_id = correlation_id or uuid4()

        if not bindings:
            return []

        # Build tasks for parallel resolution
        tasks: list[asyncio.Task[ModelBindingConfig]] = []

        for binding in bindings:
            handler_type = binding.get("handler_type")
            if not isinstance(handler_type, str):
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="resolve_many_async",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Each binding must have a 'handler_type' string field",
                    context=context,
                )

            config_ref = binding.get("config_ref")
            if config_ref is not None and not isinstance(config_ref, str):
                config_ref = None

            inline_config = binding.get("config")
            if inline_config is not None and not isinstance(inline_config, dict):
                inline_config = None

            task = asyncio.create_task(
                self.resolve_async(
                    handler_type=handler_type,
                    config_ref=config_ref,
                    inline_config=inline_config,
                    correlation_id=correlation_id,
                )
            )
            tasks.append(task)

        # Gather results - collect all exceptions for better error reporting
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and aggregate them
        failed_handler_types: list[str] = []
        configs: list[ModelBindingConfig] = []

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                handler_type = bindings[i].get("handler_type", f"binding[{i}]")
                failed_handler_types.append(str(handler_type))
                # Log the detailed error for debugging, but don't expose in exception
                # (exception message could contain sensitive config values)
                logger.debug(
                    "Configuration resolution failed for handler '%s': %s",
                    handler_type,
                    result,
                    extra={"correlation_id": str(correlation_id)},
                )
            else:
                # Type narrowing: result is ModelBindingConfig after BaseException check
                configs.append(result)

        if failed_handler_types:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="resolve_many_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Failed to resolve {len(failed_handler_types)} configuration(s) "
                f"for handlers: {', '.join(failed_handler_types)}",
                context=context,
            )

        return configs

    # === Cache Management ===

    def refresh(self, handler_type: str) -> None:
        """Invalidate cached config for a handler type.

        Args:
            handler_type: The handler type to refresh.
        """
        with self._lock:
            if handler_type in self._cache:
                del self._cache[handler_type]
                self._refreshes += 1

    def refresh_all(self) -> None:
        """Invalidate all cached configs."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._refreshes += count

    def get_cache_stats(self) -> ModelBindingConfigCacheStats:
        """Get cache statistics.

        Returns:
            ModelBindingConfigCacheStats with hit/miss/load counts and lock stats.
        """
        with self._lock:
            return ModelBindingConfigCacheStats(
                total_entries=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                refreshes=self._refreshes,
                expired_evictions=self._expired_evictions,
                lru_evictions=self._lru_evictions,
                file_loads=self._file_loads,
                env_loads=self._env_loads,
                vault_loads=self._vault_loads,
                async_key_lock_count=len(self._async_key_locks),
                async_key_lock_cleanups=self._async_key_lock_cleanups,
            )

    # === Internal Methods ===

    def _get_async_key_lock(self, handler_type: str) -> asyncio.Lock:
        """Get or create an async lock for a specific handler_type.

        Includes periodic cleanup of stale locks to prevent unbounded memory
        growth in long-running processes. Cleanup is triggered when the number
        of locks exceeds the configured threshold (async_lock_cleanup_threshold).

        Thread Safety:
            Uses threading.Lock to safely access the key locks dictionary,
            ensuring thread-safe creation of new locks. Cleanup only removes
            locks that are not currently held.

        Args:
            handler_type: The handler type to get a lock for.

        Returns:
            asyncio.Lock for the given handler_type.

        Note:
            The cleanup threshold is configurable via
            ModelBindingConfigResolverConfig.async_lock_cleanup_threshold.
            Default is 1000 locks.
        """
        with self._lock:
            # Periodic cleanup when threshold exceeded (uses config value)
            threshold = self._config.async_lock_cleanup_threshold
            if len(self._async_key_locks) > threshold:
                self._cleanup_stale_async_key_locks()

            if handler_type not in self._async_key_locks:
                self._async_key_locks[handler_type] = asyncio.Lock()
                self._async_key_lock_timestamps[handler_type] = time.monotonic()
            return self._async_key_locks[handler_type]

    def _cleanup_stale_async_key_locks(self) -> None:
        """Remove async key locks that have not been used recently.

        Only removes locks that are:
        1. Older than the configured max age (async_lock_max_age_seconds)
        2. Not currently held (not locked)

        Thread Safety:
            Must be called while holding self._lock. Safe to call from
            any thread as it only modifies internal state.

        Note:
            This method is called periodically from _get_async_key_lock()
            when the lock count exceeds the threshold. It does not require
            external scheduling.

            The max age is configurable via
            ModelBindingConfigResolverConfig.async_lock_max_age_seconds.
            Default is 3600 seconds (1 hour).
        """
        current_time = time.monotonic()
        stale_keys: list[str] = []
        max_age = self._config.async_lock_max_age_seconds

        for key, timestamp in self._async_key_lock_timestamps.items():
            age = current_time - timestamp
            if age > max_age:
                lock = self._async_key_locks.get(key)
                # Only remove locks that are not currently held
                if lock is not None and not lock.locked():
                    stale_keys.append(key)

        for key in stale_keys:
            del self._async_key_locks[key]
            del self._async_key_lock_timestamps[key]

        if stale_keys:
            self._async_key_lock_cleanups += 1
            logger.debug(
                "Cleaned up stale async key locks",
                extra={
                    "cleaned_count": len(stale_keys),
                    "remaining_count": len(self._async_key_locks),
                    "max_age_seconds": max_age,
                },
            )

    def _cleanup_async_key_lock_for_eviction(self, key: str) -> None:
        """Clean up async key lock when its associated cache entry is evicted.

        This method is called during LRU eviction or TTL expiration to ensure
        async locks don't leak when their corresponding cache entries are removed.

        Thread Safety:
            Must be called while holding self._lock. Only removes locks that
            are not currently held to prevent race conditions with concurrent
            async operations.

        Args:
            key: The handler_type key whose async lock should be cleaned up.

        Note:
            If the lock is currently held (e.g., an async operation is in progress),
            it will NOT be removed. The lock will be cleaned up later during
            periodic cleanup or when the operation completes.
        """
        if key in self._async_key_locks:
            lock = self._async_key_locks[key]
            # Only remove locks that are not currently held
            # If locked, an async operation is in progress and will need the lock
            if not lock.locked():
                del self._async_key_locks[key]
                if key in self._async_key_lock_timestamps:
                    del self._async_key_lock_timestamps[key]
                logger.debug(
                    "Cleaned up async key lock for evicted cache entry",
                    extra={"handler_type": key},
                )

    def _get_from_cache(self, handler_type: str) -> ModelBindingConfig | None:
        """Get config from cache if present and not expired.

        Args:
            handler_type: The handler type to look up.

        Returns:
            ModelBindingConfig if cached and valid, None otherwise.

        Note:
            This method does NOT increment the miss counter. Misses are counted
            at the point where resolution from sources occurs (either in
            _cache_config when caching is enabled, or in resolve/resolve_async
            when caching is disabled). This ensures accurate miss counting in
            the async path which uses a double-check locking pattern.

            When max_cache_entries is configured, this method moves accessed entries
            to the end of the OrderedDict to maintain LRU ordering. This ensures
            the least recently used entry is at the front for eviction.

            When a cache entry is evicted due to TTL expiration, this method also
            cleans up the associated async key lock to prevent memory leaks.
        """
        if not self._config.enable_caching:
            return None

        cached = self._cache.get(handler_type)
        if cached is None:
            return None

        if cached.is_expired():
            del self._cache[handler_type]
            self._expired_evictions += 1
            # Clean up the async lock for this evicted entry
            self._cleanup_async_key_lock_for_eviction(handler_type)
            return None

        # Move to end for LRU tracking (most recently used)
        # This is a no-op if max_cache_entries is None, but we do it anyway
        # for consistency since OrderedDict.move_to_end() is O(1)
        self._cache.move_to_end(handler_type)
        self._hits += 1
        return cached.config

    def _cache_config(
        self,
        handler_type: str,
        config: ModelBindingConfig,
        source: str,
    ) -> None:
        """Cache a resolved configuration with TTL.

        Args:
            handler_type: The handler type being cached.
            config: The configuration to cache.
            source: Description of the configuration source.

        Note:
            When max_cache_entries is configured and the cache is at capacity,
            this method evicts the least recently used (LRU) entry before adding
            the new one. The LRU entry is the first entry in the OrderedDict
            since entries are moved to the end on access.

            When a cache entry is evicted via LRU, this method also cleans up
            the associated async key lock to prevent memory leaks. The lock is
            only removed if it is not currently held by an async operation.

        Thread Safety:
            This method is ALWAYS called while holding self._lock (from resolve()
            or resolve_async()), ensuring that LRU eviction and cache write are
            atomic. There is no race window where another thread could observe
            an inconsistent cache state. See PR #168 review for analysis.
        """
        # Evict LRU entry if cache is at capacity (before adding new entry)
        # NOTE: This entire operation is atomic because the caller holds self._lock
        max_entries = self._config.max_cache_entries
        if max_entries is not None and handler_type not in self._cache:
            # Only evict if adding a NEW entry (not updating existing)
            while len(self._cache) >= max_entries:
                # popitem(last=False) removes the first (oldest/LRU) entry
                evicted_key, _ = self._cache.popitem(last=False)
                self._lru_evictions += 1
                # Clean up the async lock for this evicted entry
                self._cleanup_async_key_lock_for_eviction(evicted_key)
                logger.debug(
                    "LRU eviction: removed cache entry",
                    extra={
                        "evicted_handler_type": evicted_key,
                        "new_handler_type": handler_type,
                        "max_cache_entries": max_entries,
                    },
                )

        now = datetime.now(UTC)
        ttl_seconds = self._config.cache_ttl_seconds
        expires_at = now + timedelta(seconds=ttl_seconds)

        self._cache[handler_type] = ModelConfigCacheEntry(
            config=config,
            expires_at=expires_at,
            source=source,
        )
        self._misses += 1

    def _describe_source(
        self,
        config_ref: str | None,
        inline_config: dict[str, JsonType] | None,
    ) -> str:
        """Create a description of the configuration source for debugging.

        Args:
            config_ref: The config reference, if any.
            inline_config: The inline config, if any.

        Returns:
            Human-readable source description.
        """
        sources: list[str] = []
        if config_ref:
            # Don't expose full path - just scheme
            if ":" in config_ref:
                scheme = config_ref.split(":")[0]
                sources.append(f"{scheme}:...")
            else:
                sources.append("unknown")
        if inline_config:
            sources.append("inline")
        sources.append("env_overrides")
        return "+".join(sources) if sources else "default"

    def _resolve_config(
        self,
        handler_type: str,
        config_ref: str | None,
        inline_config: dict[str, JsonType] | None,
        correlation_id: UUID,
    ) -> ModelBindingConfig:
        """Resolve configuration from sources synchronously.

        Args:
            handler_type: Handler type identifier.
            config_ref: Optional external configuration reference.
            inline_config: Optional inline configuration.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Resolved ModelBindingConfig.

        Raises:
            ProtocolConfigurationError: If configuration is invalid.
        """
        # Start with empty config
        merged_config: dict[str, object] = {}

        # Load from config_ref if provided
        if config_ref:
            ref_config = self._load_from_ref(config_ref, correlation_id)
            merged_config.update(ref_config)

        # Merge inline config (takes precedence over ref)
        if inline_config:
            merged_config.update(inline_config)

        # Ensure handler_type is set
        merged_config["handler_type"] = handler_type

        # Apply environment variable overrides (highest priority)
        merged_config = self._apply_env_overrides(
            merged_config, handler_type, correlation_id
        )

        # Resolve any vault: references in the config
        merged_config = self._resolve_vault_refs(merged_config, correlation_id)

        # Validate and construct the final config
        return self._validate_config(merged_config, handler_type, correlation_id)

    async def _resolve_config_async(
        self,
        handler_type: str,
        config_ref: str | None,
        inline_config: dict[str, object] | None,
        correlation_id: UUID,
    ) -> ModelBindingConfig:
        """Resolve configuration from sources asynchronously.

        Args:
            handler_type: Handler type identifier.
            config_ref: Optional external configuration reference.
            inline_config: Optional inline configuration.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Resolved ModelBindingConfig.

        Raises:
            ProtocolConfigurationError: If configuration is invalid.
        """
        # Start with empty config
        merged_config: dict[str, object] = {}

        # Load from config_ref if provided
        if config_ref:
            ref_config = await self._load_from_ref_async(config_ref, correlation_id)
            merged_config.update(ref_config)

        # Merge inline config (takes precedence over ref)
        if inline_config:
            merged_config.update(inline_config)

        # Ensure handler_type is set
        merged_config["handler_type"] = handler_type

        # Apply environment variable overrides (highest priority)
        merged_config = self._apply_env_overrides(
            merged_config, handler_type, correlation_id
        )

        # Resolve any vault: references in the config (async)
        merged_config = await self._resolve_vault_refs_async(
            merged_config, correlation_id
        )

        # Validate and construct the final config
        return self._validate_config(merged_config, handler_type, correlation_id)

    def _load_from_ref(
        self,
        config_ref: str,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Load configuration from a config_ref.

        Args:
            config_ref: Configuration reference using scheme format (file:, env:, vault:).
                Examples: file:configs/db.yaml, env:DB_CONFIG, vault:secret/data/db#password
            correlation_id: Correlation ID for error tracking.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ProtocolConfigurationError: If reference is invalid or cannot be loaded.
        """
        # Parse the config reference
        parse_result = ModelConfigRef.parse(config_ref)
        if not parse_result:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref",
                target_name="binding_config_resolver",
            )
            # Log the detailed error for debugging, but don't expose parse details
            # in the exception message (config_ref could contain sensitive paths)
            logger.debug(
                "Config reference parsing failed: %s",
                parse_result.error_message,
                extra={"correlation_id": str(correlation_id)},
            )
            raise ProtocolConfigurationError(
                "Invalid config reference format",
                context=context,
            )

        ref = parse_result.config_ref
        if ref is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Config reference parse result has no config_ref",
                context=context,
            )

        # Check scheme is allowed
        if ref.scheme.value not in self._config.allowed_schemes:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Scheme '{ref.scheme.value}' is not in allowed schemes",
                context=context,
            )

        # Load based on scheme
        if ref.scheme == EnumConfigRefScheme.FILE:
            return self._load_from_file(Path(ref.path), correlation_id)
        elif ref.scheme == EnumConfigRefScheme.ENV:
            return self._load_from_env(ref.path, correlation_id)
        elif ref.scheme == EnumConfigRefScheme.VAULT:
            return self._load_from_vault(ref.path, ref.fragment, correlation_id)
        else:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Unsupported scheme: {ref.scheme.value}",
                context=context,
            )

    async def _load_from_ref_async(
        self,
        config_ref: str,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Load configuration from a config_ref asynchronously.

        Args:
            config_ref: Configuration reference using scheme format (file:, env:, vault:).
                Examples: file:configs/db.yaml, env:DB_CONFIG, vault:secret/data/db#password
            correlation_id: Correlation ID for error tracking.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ProtocolConfigurationError: If reference is invalid or cannot be loaded.
        """
        # Parse the config reference
        parse_result = ModelConfigRef.parse(config_ref)
        if not parse_result:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref_async",
                target_name="binding_config_resolver",
            )
            # Log the detailed error for debugging, but don't expose parse details
            # in the exception message (config_ref could contain sensitive paths)
            logger.debug(
                "Config reference parsing failed: %s",
                parse_result.error_message,
                extra={"correlation_id": str(correlation_id)},
            )
            raise ProtocolConfigurationError(
                "Invalid config reference format",
                context=context,
            )

        ref = parse_result.config_ref
        if ref is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Config reference parse result has no config_ref",
                context=context,
            )

        # Check scheme is allowed
        if ref.scheme.value not in self._config.allowed_schemes:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Scheme '{ref.scheme.value}' is not in allowed schemes",
                context=context,
            )

        # Load based on scheme
        if ref.scheme == EnumConfigRefScheme.FILE:
            return await asyncio.to_thread(
                self._load_from_file, Path(ref.path), correlation_id
            )
        elif ref.scheme == EnumConfigRefScheme.ENV:
            # Env var access is fast, no need for thread
            return self._load_from_env(ref.path, correlation_id)
        elif ref.scheme == EnumConfigRefScheme.VAULT:
            return await self._load_from_vault_async(
                ref.path, ref.fragment, correlation_id
            )
        else:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_ref_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Unsupported scheme: {ref.scheme.value}",
                context=context,
            )

    def _load_from_file(
        self,
        path: Path,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Load config from YAML or JSON file.

        Args:
            path: Path to the configuration file.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ProtocolConfigurationError: If file cannot be read or parsed.
        """
        # Resolve relative paths against config_dir
        if not path.is_absolute():
            if self._config.config_dir is not None:
                # Validate config_dir at use-time (deferred from model construction)
                try:
                    config_dir_exists = self._config.config_dir.exists()
                    config_dir_is_dir = self._config.config_dir.is_dir()
                except ValueError:
                    # config_dir contains null bytes (defense-in-depth)
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="load_from_file",
                        target_name="binding_config_resolver",
                    )
                    raise ProtocolConfigurationError(
                        "Invalid config_dir path: contains invalid characters",
                        context=context,
                    )
                if not config_dir_exists:
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="load_from_file",
                        target_name="binding_config_resolver",
                    )
                    raise ProtocolConfigurationError(
                        f"config_dir does not exist: path='{self._config.config_dir}'",
                        context=context,
                    )
                if not config_dir_is_dir:
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="load_from_file",
                        target_name="binding_config_resolver",
                    )
                    raise ProtocolConfigurationError(
                        f"config_dir exists but is not a directory: path='{self._config.config_dir}'",
                        context=context,
                    )
                path = self._config.config_dir / path
            else:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="load_from_file",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Relative path provided but no config_dir configured",
                    context=context,
                )

        # Security: Check for symlinks if not allowed
        # Check before resolve() to detect symlinks in the original path
        try:
            is_symlink = not self._config.allow_symlinks and path.is_symlink()
        except ValueError:
            # Path contains null bytes or other invalid characters
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Invalid configuration file path: contains invalid characters",
                context=context,
            )
        if is_symlink:
            logger.warning(
                "Symlink rejected in config file path",
                extra={"correlation_id": str(correlation_id)},
            )
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Configuration file symlinks not allowed",
                context=context,
            )

        # Resolve to absolute path for security validation
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError, ValueError):
            # ValueError: path contains null bytes or other invalid characters
            # OSError/RuntimeError: filesystem/symlink resolution errors
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Invalid configuration file path",
                context=context,
            )

        # Security: Check if any path component is a symlink (when symlinks disallowed)
        # This catches symlinks in parent directories (e.g., /etc/configs -> /tmp/evil)
        if not self._config.allow_symlinks:
            current = path
            while current != current.parent:
                try:
                    is_current_symlink = current.is_symlink()
                except ValueError:
                    # Path contains null bytes or other invalid characters
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="load_from_file",
                        target_name="binding_config_resolver",
                    )
                    raise ProtocolConfigurationError(
                        "Invalid configuration file path: contains invalid characters",
                        context=context,
                    )
                if is_current_symlink:
                    logger.warning(
                        "Symlink detected in path hierarchy",
                        extra={"correlation_id": str(correlation_id)},
                    )
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="load_from_file",
                        target_name="binding_config_resolver",
                    )
                    raise ProtocolConfigurationError(
                        "Configuration file path contains symlink",
                        context=context,
                    )
                current = current.parent

        # Security: Validate path is within config_dir if configured
        if self._config.config_dir is not None:
            try:
                config_dir_resolved = self._config.config_dir.resolve()
            except (OSError, RuntimeError, ValueError):
                # ValueError: config_dir contains null bytes (defense-in-depth)
                # OSError/RuntimeError: filesystem/symlink resolution errors
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="load_from_file",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Invalid config_dir path",
                    context=context,
                )
            try:
                resolved_path.relative_to(config_dir_resolved)
            except ValueError:
                # Path escapes config_dir - this is a path traversal attempt
                logger.warning(
                    "Path traversal detected in config file path",
                    extra={"correlation_id": str(correlation_id)},
                )
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="load_from_file",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Configuration file path traversal not allowed",
                    context=context,
                )

        # Read file with size limit
        try:
            with resolved_path.open("r") as f:
                content = f.read(MAX_CONFIG_FILE_SIZE + 1)
                if len(content) > MAX_CONFIG_FILE_SIZE:
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="load_from_file",
                        target_name="binding_config_resolver",
                    )
                    raise ProtocolConfigurationError(
                        "Configuration file exceeds size limit",
                        context=context,
                    )
        except FileNotFoundError:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Configuration file not found",
                context=context,
            )
        except IsADirectoryError:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Configuration path is a directory, not a file",
                context=context,
            )
        except PermissionError:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Permission denied reading configuration file",
                context=context,
            )
        except OSError:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "OS error reading configuration file",
                context=context,
            )

        # Parse based on extension
        suffix = resolved_path.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(content)
            elif suffix == ".json":
                data = json.loads(content)
            else:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="load_from_file",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    f"Unsupported configuration file format: {suffix}",
                    context=context,
                )
        except yaml.YAMLError:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Invalid YAML in configuration file",
                context=context,
            )
        except json.JSONDecodeError:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Invalid JSON in configuration file",
                context=context,
            )

        if not isinstance(data, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_file",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Configuration file must contain a dictionary",
                context=context,
            )

        with self._lock:
            self._file_loads += 1

        return data

    def _load_from_env(
        self,
        env_var: str,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Load config from environment variable (JSON or YAML).

        Args:
            env_var: Environment variable name containing configuration.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ProtocolConfigurationError: If env var is missing or contains invalid data.
        """
        value = os.environ.get(env_var)
        if value is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_env",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Environment variable not set: {env_var}",
                context=context,
            )

        # Try JSON first, then YAML
        data: object = None
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            try:
                data = yaml.safe_load(value)
            except yaml.YAMLError:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="load_from_env",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    f"Environment variable {env_var} contains invalid JSON/YAML",
                    context=context,
                )

        if not isinstance(data, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_env",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Environment variable {env_var} must contain a dictionary",
                context=context,
            )

        with self._lock:
            self._env_loads += 1

        return data

    def _load_from_vault(
        self,
        vault_path: str,
        fragment: str | None,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Load config from Vault secret.

        Args:
            vault_path: Vault secret path.
            fragment: Optional field within the secret.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ProtocolConfigurationError: If Vault is not configured or secret cannot be read.
        """
        secret_resolver = self._get_secret_resolver()
        if secret_resolver is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_vault",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Vault scheme used but no SecretResolver configured",
                context=context,
            )

        # Build logical name for secret resolver
        logical_name = vault_path
        if fragment:
            logical_name = f"{vault_path}#{fragment}"

        try:
            secret = secret_resolver.get_secret(logical_name, required=True)
        except (SecretResolutionError, NotImplementedError) as e:
            # SecretResolutionError: secret not found or resolution failed
            # NotImplementedError: Vault integration not yet implemented
            # SECURITY: Log at DEBUG level only - exception may contain vault paths
            logger.debug(
                "Vault configuration retrieval failed (correlation_id=%s): %s",
                correlation_id,
                e,
                extra={"correlation_id": str(correlation_id)},
            )
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault",
                target_name="binding_config_resolver",
            )
            # SECURITY: Do NOT chain original exception (from e) - it may
            # contain vault paths in its message.
            raise ProtocolConfigurationError(
                f"Failed to retrieve configuration from Vault. "
                f"correlation_id={correlation_id}",
                context=context,
            )

        if secret is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Vault secret not found",
                context=context,
            )

        # Parse secret value as JSON or YAML
        secret_value = secret.get_secret_value()
        data: object = None
        try:
            data = json.loads(secret_value)
        except json.JSONDecodeError:
            try:
                data = yaml.safe_load(secret_value)
            except yaml.YAMLError:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="load_from_vault",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Vault secret contains invalid JSON/YAML",
                    context=context,
                )

        if not isinstance(data, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Vault secret must contain a dictionary",
                context=context,
            )

        with self._lock:
            self._vault_loads += 1

        return data

    async def _load_from_vault_async(
        self,
        vault_path: str,
        fragment: str | None,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Load config from Vault secret asynchronously.

        Args:
            vault_path: Vault secret path.
            fragment: Optional field within the secret.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ProtocolConfigurationError: If Vault is not configured or secret cannot be read.
        """
        secret_resolver = self._get_secret_resolver()
        if secret_resolver is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="load_from_vault_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Vault scheme used but no SecretResolver configured",
                context=context,
            )

        # Build logical name for secret resolver
        logical_name = vault_path
        if fragment:
            logical_name = f"{vault_path}#{fragment}"

        try:
            secret = await secret_resolver.get_secret_async(logical_name, required=True)
        except (SecretResolutionError, NotImplementedError) as e:
            # SecretResolutionError: secret not found or resolution failed
            # NotImplementedError: Vault integration not yet implemented
            # SECURITY: Log at DEBUG level only - exception may contain vault paths
            logger.debug(
                "Vault configuration retrieval failed async (correlation_id=%s): %s",
                correlation_id,
                e,
                extra={"correlation_id": str(correlation_id)},
            )
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault_async",
                target_name="binding_config_resolver",
            )
            # SECURITY: Do NOT chain original exception (from e) - it may
            # contain vault paths in its message.
            raise ProtocolConfigurationError(
                f"Failed to retrieve configuration from Vault. "
                f"correlation_id={correlation_id}",
                context=context,
            )

        if secret is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Vault secret not found",
                context=context,
            )

        # Parse secret value as JSON or YAML
        secret_value = secret.get_secret_value()
        data: object = None
        try:
            data = json.loads(secret_value)
        except json.JSONDecodeError:
            try:
                data = yaml.safe_load(secret_value)
            except yaml.YAMLError:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.VAULT,
                    operation="load_from_vault_async",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Vault secret contains invalid JSON/YAML",
                    context=context,
                )

        if not isinstance(data, dict):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Vault secret must contain a dictionary",
                context=context,
            )

        with self._lock:
            self._vault_loads += 1

        return data

    def _get_secret_resolver(self) -> SecretResolver | None:
        """Get the container-resolved SecretResolver instance.

        The SecretResolver is resolved from the container during __init__.
        This method provides access to the cached instance.

        Returns:
            SecretResolver if registered in container, None otherwise.
        """
        return self._secret_resolver

    def _parse_vault_reference(self, value: str) -> tuple[str, str | None]:
        """Parse a vault: reference string into path and optional fragment.

        Extracts the vault path and optional fragment from a vault reference.
        The fragment is specified after a '#' character in the reference.

        Args:
            value: The vault reference string (e.g., "vault:secret/path#field").

        Returns:
            Tuple of (vault_path, fragment) where fragment may be None.
            The vault_path has the "vault:" prefix removed.

        Example:
            >>> self._parse_vault_reference("vault:secret/data/db#password")
            ("secret/data/db", "password")
            >>> self._parse_vault_reference("vault:secret/data/config")
            ("secret/data/config", None)
        """
        vault_path = value[6:]  # Remove "vault:" prefix
        return _split_path_and_fragment(vault_path)

    def _has_vault_references(self, config: dict[str, object]) -> bool:
        """Check if config contains any vault: references (including nested dicts and lists).

        Recursively scans the configuration dictionary to detect any string
        values that start with "vault:".

        Args:
            config: Configuration dictionary to check.

        Returns:
            True if any vault: references are found, False otherwise.
        """
        for value in config.values():
            if isinstance(value, str) and value.startswith("vault:"):
                return True
            if isinstance(value, dict):
                if self._has_vault_references(value):
                    return True
            if isinstance(value, list):
                if self._has_vault_references_in_list(value):
                    return True
        return False

    def _has_vault_references_in_list(self, items: list[object]) -> bool:
        """Check if a list contains any vault: references (including nested structures).

        Args:
            items: List to check for vault references.

        Returns:
            True if any vault: references are found, False otherwise.
        """
        for item in items:
            if isinstance(item, str) and item.startswith("vault:"):
                return True
            if isinstance(item, dict):
                if self._has_vault_references(item):
                    return True
            if isinstance(item, list):
                if self._has_vault_references_in_list(item):
                    return True
        return False

    def _apply_env_overrides(
        self,
        config: dict[str, object],
        handler_type: str,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Apply environment variable overrides.

        Environment variables follow the pattern:
        {env_prefix}_{HANDLER_TYPE}_{FIELD}

        For example: HANDLER_DB_TIMEOUT_MS=10000

        Args:
            config: Base configuration dictionary.
            handler_type: Handler type for env var name construction.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Configuration with environment overrides applied.
        """
        result = dict(config)
        prefix = self._config.env_prefix
        handler_upper = handler_type.upper()

        # Track retry policy overrides separately
        retry_overrides: dict[str, object] = {}

        for env_field, model_field in _ENV_OVERRIDE_FIELDS.items():
            env_name = f"{prefix}_{handler_upper}_{env_field}"
            env_value = os.environ.get(env_name)

            if env_value is not None:
                # Convert value based on expected type
                converted = self._convert_env_value(
                    env_value, model_field, env_name, correlation_id
                )
                if converted is not None:
                    if env_field in _RETRY_POLICY_FIELDS:
                        retry_overrides[model_field] = converted
                    else:
                        result[model_field] = converted

        # Merge retry policy overrides if any
        if retry_overrides:
            existing_retry = result.get("retry_policy")
            if isinstance(existing_retry, dict):
                merged_retry = dict(existing_retry)
                merged_retry.update(retry_overrides)
                result["retry_policy"] = merged_retry
            elif isinstance(existing_retry, ModelRetryPolicy):
                # Convert to dict, update, leave as dict for later construction
                merged_retry = existing_retry.model_dump()
                merged_retry.update(retry_overrides)
                result["retry_policy"] = merged_retry
            else:
                result["retry_policy"] = retry_overrides

        return result

    def _convert_env_value(
        self,
        value: str,
        field: str,
        env_name: str,
        correlation_id: UUID,
    ) -> object | None:
        """Convert environment variable string to appropriate type.

        This method handles type coercion for environment variable overrides.
        The behavior on invalid values depends on the ``strict_env_coercion``
        configuration setting:

        **Strict mode** (``strict_env_coercion=True``):
            Raises ``ProtocolConfigurationError`` immediately when a value
            cannot be converted to the expected type. This is appropriate
            for production environments where configuration errors should
            fail fast.

        **Non-strict mode** (``strict_env_coercion=False``, default):
            Logs a warning and returns ``None``. When ``None`` is returned,
            the calling code skips the override entirely, leaving the
            original configuration value unchanged. This is intentional
            conservative behavior: rather than applying a potentially
            incorrect default, the system preserves the existing
            configuration when an environment variable contains an
            invalid value.

        Args:
            value: String value from environment.
            field: Field name to determine type.
            env_name: Full environment variable name for error messages.
            correlation_id: Correlation ID for error tracking.

        Returns:
            The converted value if conversion succeeds, or ``None`` if
            conversion fails in non-strict mode. When ``None`` is returned,
            the override is skipped and the original configuration value
            is preserved (the invalid environment variable is not applied).

        Raises:
            ProtocolConfigurationError: If ``strict_env_coercion`` is enabled
                and the value cannot be converted to the expected type.

        Example:
            Boolean coercion accepts case-insensitive values::

                # All these evaluate to True:
                # HANDLER_DB_ENABLED=true, HANDLER_DB_ENABLED=1
                # HANDLER_DB_ENABLED=yes, HANDLER_DB_ENABLED=on

                # All these evaluate to False:
                # HANDLER_DB_ENABLED=false, HANDLER_DB_ENABLED=0
                # HANDLER_DB_ENABLED=no, HANDLER_DB_ENABLED=off

            Integer and float fields accept standard numeric strings::

                # Integer: HANDLER_DB_TIMEOUT_MS=5000
                # Float: HANDLER_DB_RATE_LIMIT_PER_SECOND=100.5

            Invalid values in non-strict mode are skipped (original preserved)::

                # HANDLER_DB_ENABLED=invalid -> warning logged, original kept
                # HANDLER_DB_TIMEOUT_MS=not_a_number -> warning logged, original kept
        """
        # Boolean fields
        if field == "enabled":
            valid_true = {"true", "1", "yes", "on"}
            valid_false = {"false", "0", "no", "off"}
            value_lower = value.lower()

            if value_lower in valid_true:
                return True
            if value_lower in valid_false:
                return False

            # Invalid boolean value - handle based on strict mode
            self._handle_conversion_error(
                env_name=env_name,
                field=field,
                expected_type="boolean (true/false/1/0/yes/no/on/off)",
                correlation_id=correlation_id,
            )
            # In non-strict mode, skip override (return None) to match other types
            return None

        # Integer fields
        if field in {
            "priority",
            "timeout_ms",
            "max_retries",
            "base_delay_ms",
            "max_delay_ms",
        }:
            try:
                return int(value)
            except ValueError:
                self._handle_conversion_error(
                    env_name=env_name,
                    field=field,
                    expected_type="integer",
                    correlation_id=correlation_id,
                )
                return None

        # Float fields
        if field == "rate_limit_per_second":
            try:
                return float(value)
            except ValueError:
                self._handle_conversion_error(
                    env_name=env_name,
                    field=field,
                    expected_type="float",
                    correlation_id=correlation_id,
                )
                return None

        # String fields
        if field in {"name", "backoff_strategy"}:
            return value

        return value

    def _handle_conversion_error(
        self,
        env_name: str,
        field: str,
        expected_type: str,
        correlation_id: UUID,
    ) -> None:
        """Handle type conversion error based on strict_env_coercion setting.

        In strict mode, raises ProtocolConfigurationError.
        In lenient mode, logs a warning with structured context.

        Args:
            env_name: Full environment variable name.
            field: Field name that was being set.
            expected_type: Expected type name (e.g., "integer", "float").
            correlation_id: Correlation ID for error tracking.

        Raises:
            ProtocolConfigurationError: If strict_env_coercion is enabled.
        """
        if self._config.strict_env_coercion:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="convert_env_value",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Invalid {expected_type} value in environment variable "
                f"'{env_name}' for field '{field}'",
                context=context,
            )

        logger.warning(
            "Invalid %s value in environment variable '%s' for field '%s'; "
            "override will be skipped",
            expected_type,
            env_name,
            field,
            extra={
                "correlation_id": str(correlation_id),
                "env_var": env_name,
                "field": field,
                "expected_type": expected_type,
            },
        )

    def _resolve_vault_refs(
        self,
        config: dict[str, object],
        correlation_id: UUID,
        depth: int = 0,
    ) -> dict[str, object]:
        """Resolve any vault: references in config values.

        Scans all string values for vault: prefix and resolves them
        using the SecretResolver.

        Args:
            config: Configuration dictionary.
            correlation_id: Correlation ID for error tracking.
            depth: Current recursion depth (default 0).

        Returns:
            Configuration with vault references resolved.

        Raises:
            ProtocolConfigurationError: If recursion depth exceeds maximum,
                or if fail_on_vault_error is True and a vault reference fails.
        """
        if depth > _MAX_NESTED_CONFIG_DEPTH:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="resolve_vault_refs",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Configuration nesting exceeds maximum depth of {_MAX_NESTED_CONFIG_DEPTH}",
                context=context,
            )

        secret_resolver = self._get_secret_resolver()
        if secret_resolver is None:
            # Check if there are vault references that need resolution
            # If fail_on_vault_error is True and vault refs exist, this is a security issue
            if self._config.fail_on_vault_error and self._has_vault_references(config):
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="resolve_vault_refs",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Config contains vault: references but no SecretResolver is configured",
                    context=context,
                )
            return config

        result: dict[str, object] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("vault:"):
                # Parse vault reference using helper method
                vault_path, fragment = self._parse_vault_reference(value)
                logical_name = f"{vault_path}#{fragment}" if fragment else vault_path

                try:
                    secret = secret_resolver.get_secret(logical_name, required=False)
                    if secret is not None:
                        result[key] = secret.get_secret_value()
                    else:
                        # Secret not found - check fail_on_vault_error
                        if self._config.fail_on_vault_error:
                            logger.error(
                                "Vault secret not found for config key '%s'",
                                key,
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "config_key": key,
                                },
                            )
                            context = ModelInfraErrorContext.with_correlation(
                                correlation_id=correlation_id,
                                transport_type=EnumInfraTransportType.VAULT,
                                operation="resolve_vault_refs",
                                target_name="binding_config_resolver",
                            )
                            raise ProtocolConfigurationError(
                                f"Vault secret not found for config key '{key}'",
                                context=context,
                            )
                        result[key] = value  # Keep original if not found
                except (SecretResolutionError, NotImplementedError) as e:
                    # SecretResolutionError: secret not found or resolution failed
                    # NotImplementedError: Vault integration not yet implemented
                    # SECURITY: Log at DEBUG level only - exception may contain vault paths
                    # Use DEBUG to capture details for troubleshooting without exposing
                    # sensitive paths in production logs
                    logger.debug(
                        "Vault resolution failed for config key '%s' "
                        "(correlation_id=%s): %s",
                        key,
                        correlation_id,
                        e,
                        extra={
                            "correlation_id": str(correlation_id),
                            "config_key": key,
                        },
                    )
                    # Respect fail_on_vault_error config option
                    if self._config.fail_on_vault_error:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.VAULT,
                            operation="resolve_vault_refs",
                            target_name="binding_config_resolver",
                        )
                        # SECURITY: Do NOT chain original exception (from e) - it may
                        # contain vault paths in its message. Original error is logged
                        # at DEBUG level above for troubleshooting.
                        raise ProtocolConfigurationError(
                            f"Failed to resolve Vault secret reference for config key "
                            f"'{key}'. correlation_id={correlation_id}",
                            context=context,
                        )
                    # Keep original on error (may be insecure - logged above)
                    result[key] = value
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                result[key] = self._resolve_vault_refs(value, correlation_id, depth + 1)
            elif isinstance(value, list):
                # Recursively resolve vault references in list items
                result[key] = self._resolve_vault_refs_in_list(
                    value, secret_resolver, correlation_id, depth + 1
                )
            else:
                result[key] = value

        return result

    def _resolve_vault_refs_in_list(
        self,
        items: list[object],
        secret_resolver: SecretResolver,
        correlation_id: UUID,
        depth: int,
    ) -> list[object]:
        """Resolve vault: references within a list.

        Processes each item in the list, resolving any vault: references found
        in strings, nested dicts, or nested lists.

        Args:
            items: List of items to process.
            secret_resolver: SecretResolver instance for vault lookups.
            correlation_id: Correlation ID for error tracking.
            depth: Current recursion depth.

        Returns:
            List with vault references resolved.

        Raises:
            ProtocolConfigurationError: If recursion depth exceeds maximum,
                or if fail_on_vault_error is True and a vault reference fails.
        """
        if depth > _MAX_NESTED_CONFIG_DEPTH:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="resolve_vault_refs_in_list",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Configuration nesting exceeds maximum depth of {_MAX_NESTED_CONFIG_DEPTH}",
                context=context,
            )

        result: list[object] = []
        for i, item in enumerate(items):
            if isinstance(item, str) and item.startswith("vault:"):
                # Parse vault reference using helper method
                vault_path, fragment = self._parse_vault_reference(item)
                logical_name = f"{vault_path}#{fragment}" if fragment else vault_path

                try:
                    secret = secret_resolver.get_secret(logical_name, required=False)
                    if secret is not None:
                        result.append(secret.get_secret_value())
                    else:
                        # Secret not found - check fail_on_vault_error
                        if self._config.fail_on_vault_error:
                            logger.error(
                                "Vault secret not found at list index %d",
                                i,
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "list_index": i,
                                },
                            )
                            context = ModelInfraErrorContext.with_correlation(
                                correlation_id=correlation_id,
                                transport_type=EnumInfraTransportType.VAULT,
                                operation="resolve_vault_refs_in_list",
                                target_name="binding_config_resolver",
                            )
                            raise ProtocolConfigurationError(
                                f"Vault secret not found at list index {i}",
                                context=context,
                            )
                        result.append(item)  # Keep original if not found
                except (SecretResolutionError, NotImplementedError) as e:
                    # SecretResolutionError: secret not found or resolution failed
                    # NotImplementedError: Vault integration not yet implemented
                    # SECURITY: Log at DEBUG level only - exception may contain vault paths
                    # Do not log vault_path - reveals secret structure
                    logger.debug(
                        "Vault resolution failed at list index %d "
                        "(correlation_id=%s): %s",
                        i,
                        correlation_id,
                        e,
                        extra={
                            "correlation_id": str(correlation_id),
                            "list_index": i,
                        },
                    )
                    if self._config.fail_on_vault_error:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.VAULT,
                            operation="resolve_vault_refs_in_list",
                            target_name="binding_config_resolver",
                        )
                        # SECURITY: Do NOT chain original exception (from e) - it may
                        # contain vault paths in its message.
                        raise ProtocolConfigurationError(
                            f"Failed to resolve Vault secret reference at list index {i}. "
                            f"correlation_id={correlation_id}",
                            context=context,
                        )
                    result.append(item)
            elif isinstance(item, dict):
                result.append(self._resolve_vault_refs(item, correlation_id, depth + 1))
            elif isinstance(item, list):
                result.append(
                    self._resolve_vault_refs_in_list(
                        item, secret_resolver, correlation_id, depth + 1
                    )
                )
            else:
                result.append(item)

        return result

    async def _resolve_vault_refs_async(
        self,
        config: dict[str, object],
        correlation_id: UUID,
        depth: int = 0,
    ) -> dict[str, object]:
        """Resolve any vault: references in config values asynchronously.

        Args:
            config: Configuration dictionary.
            correlation_id: Correlation ID for error tracking.
            depth: Current recursion depth (default 0).

        Returns:
            Configuration with vault references resolved.

        Raises:
            ProtocolConfigurationError: If recursion depth exceeds maximum,
                or if fail_on_vault_error is True and a vault reference fails.
        """
        if depth > _MAX_NESTED_CONFIG_DEPTH:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="resolve_vault_refs_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Configuration nesting exceeds maximum depth of {_MAX_NESTED_CONFIG_DEPTH}",
                context=context,
            )

        secret_resolver = self._get_secret_resolver()
        if secret_resolver is None:
            # Check if there are vault references that need resolution
            # If fail_on_vault_error is True and vault refs exist, this is a security issue
            if self._config.fail_on_vault_error and self._has_vault_references(config):
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="resolve_vault_refs_async",
                    target_name="binding_config_resolver",
                )
                raise ProtocolConfigurationError(
                    "Config contains vault: references but no SecretResolver is configured",
                    context=context,
                )
            return config

        result: dict[str, object] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("vault:"):
                # Parse vault reference using helper method
                vault_path, fragment = self._parse_vault_reference(value)
                logical_name = f"{vault_path}#{fragment}" if fragment else vault_path

                try:
                    secret = await secret_resolver.get_secret_async(
                        logical_name, required=False
                    )
                    if secret is not None:
                        result[key] = secret.get_secret_value()
                    else:
                        # Secret not found - check fail_on_vault_error
                        if self._config.fail_on_vault_error:
                            logger.error(
                                "Vault secret not found for config key '%s'",
                                key,
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "config_key": key,
                                },
                            )
                            context = ModelInfraErrorContext.with_correlation(
                                correlation_id=correlation_id,
                                transport_type=EnumInfraTransportType.VAULT,
                                operation="resolve_vault_refs_async",
                                target_name="binding_config_resolver",
                            )
                            raise ProtocolConfigurationError(
                                f"Vault secret not found for config key '{key}'",
                                context=context,
                            )
                        result[key] = value  # Keep original if not found
                except (SecretResolutionError, NotImplementedError) as e:
                    # SecretResolutionError: secret not found or resolution failed
                    # NotImplementedError: Vault integration not yet implemented
                    # SECURITY: Log at DEBUG level only - exception may contain vault paths
                    # Use DEBUG to capture details for troubleshooting without exposing
                    # sensitive paths in production logs
                    logger.debug(
                        "Vault resolution failed for config key '%s' "
                        "(correlation_id=%s): %s",
                        key,
                        correlation_id,
                        e,
                        extra={
                            "correlation_id": str(correlation_id),
                            "config_key": key,
                        },
                    )
                    # Respect fail_on_vault_error config option
                    if self._config.fail_on_vault_error:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.VAULT,
                            operation="resolve_vault_refs_async",
                            target_name="binding_config_resolver",
                        )
                        # SECURITY: Do NOT chain original exception (from e) - it may
                        # contain vault paths in its message. Original error is logged
                        # at DEBUG level above for troubleshooting.
                        raise ProtocolConfigurationError(
                            f"Failed to resolve Vault secret reference for config key "
                            f"'{key}'. correlation_id={correlation_id}",
                            context=context,
                        )
                    # Keep original on error (may be insecure - logged above)
                    result[key] = value
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                result[key] = await self._resolve_vault_refs_async(
                    value, correlation_id, depth + 1
                )
            elif isinstance(value, list):
                # Recursively resolve vault references in list items
                result[key] = await self._resolve_vault_refs_in_list_async(
                    value, secret_resolver, correlation_id, depth + 1
                )
            else:
                result[key] = value

        return result

    async def _resolve_vault_refs_in_list_async(
        self,
        items: list[object],
        secret_resolver: SecretResolver,
        correlation_id: UUID,
        depth: int,
    ) -> list[object]:
        """Resolve vault: references within a list asynchronously.

        Processes each item in the list, resolving any vault: references found
        in strings, nested dicts, or nested lists.

        Args:
            items: List of items to process.
            secret_resolver: SecretResolver instance for vault lookups.
            correlation_id: Correlation ID for error tracking.
            depth: Current recursion depth.

        Returns:
            List with vault references resolved.

        Raises:
            ProtocolConfigurationError: If recursion depth exceeds maximum,
                or if fail_on_vault_error is True and a vault reference fails.
        """
        if depth > _MAX_NESTED_CONFIG_DEPTH:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="resolve_vault_refs_in_list_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                f"Configuration nesting exceeds maximum depth of {_MAX_NESTED_CONFIG_DEPTH}",
                context=context,
            )

        result: list[object] = []
        for i, item in enumerate(items):
            if isinstance(item, str) and item.startswith("vault:"):
                # Parse vault reference using helper method
                vault_path, fragment = self._parse_vault_reference(item)
                logical_name = f"{vault_path}#{fragment}" if fragment else vault_path

                try:
                    secret = await secret_resolver.get_secret_async(
                        logical_name, required=False
                    )
                    if secret is not None:
                        result.append(secret.get_secret_value())
                    else:
                        # Secret not found - check fail_on_vault_error
                        if self._config.fail_on_vault_error:
                            logger.error(
                                "Vault secret not found at list index %d",
                                i,
                                extra={
                                    "correlation_id": str(correlation_id),
                                    "list_index": i,
                                },
                            )
                            context = ModelInfraErrorContext.with_correlation(
                                correlation_id=correlation_id,
                                transport_type=EnumInfraTransportType.VAULT,
                                operation="resolve_vault_refs_in_list_async",
                                target_name="binding_config_resolver",
                            )
                            raise ProtocolConfigurationError(
                                f"Vault secret not found at list index {i}",
                                context=context,
                            )
                        result.append(item)  # Keep original if not found
                except (SecretResolutionError, NotImplementedError) as e:
                    # SecretResolutionError: secret not found or resolution failed
                    # NotImplementedError: Vault integration not yet implemented
                    # SECURITY: Log at DEBUG level only - exception may contain vault paths
                    # Do not log vault_path - reveals secret structure
                    logger.debug(
                        "Vault resolution failed at list index %d "
                        "(correlation_id=%s): %s",
                        i,
                        correlation_id,
                        e,
                        extra={
                            "correlation_id": str(correlation_id),
                            "list_index": i,
                        },
                    )
                    if self._config.fail_on_vault_error:
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.VAULT,
                            operation="resolve_vault_refs_in_list_async",
                            target_name="binding_config_resolver",
                        )
                        # SECURITY: Do NOT chain original exception (from e) - it may
                        # contain vault paths in its message.
                        raise ProtocolConfigurationError(
                            f"Failed to resolve Vault secret reference at list index {i}. "
                            f"correlation_id={correlation_id}",
                            context=context,
                        )
                    result.append(item)
            elif isinstance(item, dict):
                result.append(
                    await self._resolve_vault_refs_async(
                        item, correlation_id, depth + 1
                    )
                )
            elif isinstance(item, list):
                result.append(
                    await self._resolve_vault_refs_in_list_async(
                        item, secret_resolver, correlation_id, depth + 1
                    )
                )
            else:
                result.append(item)

        return result

    def _validate_config(
        self,
        config: dict[str, object],
        handler_type: str,
        correlation_id: UUID,
    ) -> ModelBindingConfig:
        """Validate and construct the final config model.

        Args:
            config: Merged configuration dictionary.
            handler_type: Handler type identifier.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Validated ModelBindingConfig.

        Raises:
            ProtocolConfigurationError: If configuration is invalid.
        """
        # Handle retry_policy construction if it's a dict
        retry_policy = config.get("retry_policy")
        if isinstance(retry_policy, dict):
            try:
                config["retry_policy"] = ModelRetryPolicy.model_validate(retry_policy)
            except ValidationError as e:
                # ValidationError: Pydantic model validation failed
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_config",
                    target_name=f"handler:{handler_type}",
                )
                # SECURITY: Log at DEBUG level only - validation error may contain
                # config values which could include secrets
                logger.debug(
                    "Retry policy validation failed for handler '%s' "
                    "(correlation_id=%s): %s",
                    handler_type,
                    correlation_id,
                    e,
                    extra={"correlation_id": str(correlation_id)},
                )
                # SECURITY: Do NOT chain original exception (from e) - Pydantic
                # validation errors may contain config values in their message.
                raise ProtocolConfigurationError(
                    f"Invalid retry policy configuration for handler '{handler_type}'. "
                    f"correlation_id={correlation_id}",
                    context=context,
                )

        # Filter to only known fields if strict validation is disabled
        if not self._config.strict_validation:
            known_fields = set(ModelBindingConfig.model_fields.keys())
            config = {k: v for k, v in config.items() if k in known_fields}

        try:
            return ModelBindingConfig.model_validate(config)
        except ValidationError as e:
            # ValidationError: Pydantic model validation failed
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_config",
                target_name=f"handler:{handler_type}",
            )
            # SECURITY: Log at DEBUG level only - validation error may contain
            # config values which could include secrets or sensitive data
            logger.debug(
                "Handler configuration validation failed for '%s' "
                "(correlation_id=%s): %s",
                handler_type,
                correlation_id,
                e,
                extra={"correlation_id": str(correlation_id)},
            )
            # SECURITY: Do NOT chain original exception (from e) - Pydantic
            # validation errors may contain config values in their message.
            raise ProtocolConfigurationError(
                f"Invalid handler configuration for type '{handler_type}'. "
                f"correlation_id={correlation_id}",
                context=context,
            )


__all__: Final[list[str]] = ["BindingConfigResolver"]
