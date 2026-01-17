# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding configuration resolver for ONEX infrastructure.

BindingConfigResolver provides a unified interface for resolving handler configurations
from multiple sources with proper priority ordering:

    1. Environment variables (HANDLER_{TYPE}_{FIELD}) - highest priority
    2. Vault secrets (via SecretResolver)
    3. File configs (YAML/JSON)
    4. Contract YAML inline config - lowest priority

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
        await container.service_registry.register_instance(
            interface=ModelBindingConfigResolverConfig,
            instance=config,
            scope="global",
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
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

# Async key lock cleanup configuration
# Prevents unbounded growth of _async_key_locks dict in long-running processes
_ASYNC_KEY_LOCK_CLEANUP_THRESHOLD: Final[int] = (
    1000  # Trigger cleanup when > 1000 locks
)
_ASYNC_KEY_LOCK_MAX_AGE_SECONDS: Final[float] = 3600.0  # Clean locks older than 1 hour


class BindingConfigResolver:  # ONEX_EXCLUDE: method_count - follows SecretResolver pattern
    """Resolver that normalizes handler configs from multiple sources.

    The BindingConfigResolver provides a unified interface for resolving handler
    configurations with proper priority ordering and caching support.

    Resolution Order:
        1. Check cache (if enabled and not expired)
        2. Parse config_ref if present (file://, env:, vault:)
        3. Load base config (from ref or inline)
        4. Apply environment variable overrides
        5. Resolve any vault:// references for secrets
        6. Validate and construct ModelBindingConfig

    Thread Safety:
        This class is thread-safe for concurrent access from both sync and
        async contexts. See module docstring for details on the locking strategy.

    Example:
        >>> # Container setup (async context required)
        >>> container = ModelONEXContainer()
        >>> config = ModelBindingConfigResolverConfig(env_prefix="HANDLER")
        >>> await container.service_registry.register_instance(
        ...     interface=ModelBindingConfigResolverConfig,
        ...     instance=config,
        ...     scope="global",
        ... )
        >>> resolver = BindingConfigResolver(container)
        >>> binding = resolver.resolve(
        ...     handler_type="db",
        ...     inline_config={"pool_size": 10}
        ... )
    """

    def __init__(
        self,
        container: ModelONEXContainer,
    ) -> None:
        """Initialize BindingConfigResolver with container-based dependency injection.

        Follows ONEX mandatory container injection pattern per CLAUDE.md.
        Config is resolved from container's service registry, and SecretResolver
        is resolved as an optional dependency.

        Args:
            container: ONEX container for dependency resolution.

        Raises:
            RuntimeError: If ModelBindingConfigResolverConfig is not registered
                in the container's service registry.
        """
        self._container = container

        # Resolve config from container's service registry
        try:
            self._config: ModelBindingConfigResolverConfig = (
                container.service_registry.resolve_service(
                    ModelBindingConfigResolverConfig
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve ModelBindingConfigResolverConfig from container. "
                f"Ensure config is registered via container.service_registry.register_instance(). "
                f"Original error: {e}"
            ) from e

        # Resolve SecretResolver from container (optional dependency)
        # This replaces the config.secret_resolver pattern
        self._secret_resolver: SecretResolver | None = None
        try:
            from omnibase_infra.runtime.secret_resolver import SecretResolver

            self._secret_resolver = container.service_registry.resolve_service(
                SecretResolver
            )
        except Exception:
            # SecretResolver is optional - if not registered, vault:// schemes won't work
            pass

        self._cache: dict[str, ModelConfigCacheEntry] = {}
        # Track mutable stats internally since ModelBindingConfigCacheStats is frozen
        self._hits = 0
        self._misses = 0
        self._expired_evictions = 0
        self._refreshes = 0
        self._file_loads = 0
        self._env_loads = 0
        self._vault_loads = 0
        self._async_key_lock_cleanups = 0  # Track cleanup events for observability

        # RLock is required for reentrant locking in the sync path:
        # resolve() holds the lock while calling _resolve_config() -> _load_from_file(),
        # which also acquires the lock to update _file_loads counter.
        # Regular Lock would cause deadlock in this scenario.
        self._lock = threading.RLock()

        # Per-key async locks to allow parallel fetches for different handler types
        # while preventing duplicate fetches for the same handler type.
        # Timestamps track when each lock was created for periodic cleanup.
        self._async_key_locks: dict[str, asyncio.Lock] = {}
        self._async_key_lock_timestamps: dict[str, float] = {}

    # === Primary API (Sync) ===

    def resolve(
        self,
        handler_type: str,
        config_ref: str | None = None,
        inline_config: dict[str, object] | None = None,
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
                Supported schemes: file://, env:, vault:
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

            return result

    def resolve_many(
        self,
        bindings: list[dict[str, object]],
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
        inline_config: dict[str, object] | None = None,
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
                inline_config=inline_config,
                correlation_id=correlation_id,
            )

            # Cache the result if caching is enabled
            if self._config.enable_caching:
                with self._lock:
                    if handler_type not in self._cache:
                        source = self._describe_source(config_ref, inline_config)
                        self._cache_config(handler_type, result, source)

            return result

    async def resolve_many_async(
        self,
        bindings: list[dict[str, object]],
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

        # Gather results - raises on first failure
        return list(await asyncio.gather(*tasks))

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
        of locks exceeds _ASYNC_KEY_LOCK_CLEANUP_THRESHOLD.

        Thread Safety:
            Uses threading.Lock to safely access the key locks dictionary,
            ensuring thread-safe creation of new locks. Cleanup only removes
            locks that are not currently held.

        Args:
            handler_type: The handler type to get a lock for.

        Returns:
            asyncio.Lock for the given handler_type.
        """
        with self._lock:
            # Periodic cleanup when threshold exceeded
            if len(self._async_key_locks) > _ASYNC_KEY_LOCK_CLEANUP_THRESHOLD:
                self._cleanup_stale_async_key_locks()

            if handler_type not in self._async_key_locks:
                self._async_key_locks[handler_type] = asyncio.Lock()
                self._async_key_lock_timestamps[handler_type] = time.monotonic()
            return self._async_key_locks[handler_type]

    def _cleanup_stale_async_key_locks(self) -> None:
        """Remove async key locks that have not been used recently.

        Only removes locks that are:
        1. Older than _ASYNC_KEY_LOCK_MAX_AGE_SECONDS
        2. Not currently held (not locked)

        Thread Safety:
            Must be called while holding self._lock. Safe to call from
            any thread as it only modifies internal state.

        Note:
            This method is called periodically from _get_async_key_lock()
            when the lock count exceeds the threshold. It does not require
            external scheduling.
        """
        current_time = time.monotonic()
        stale_keys: list[str] = []

        for key, timestamp in self._async_key_lock_timestamps.items():
            age = current_time - timestamp
            if age > _ASYNC_KEY_LOCK_MAX_AGE_SECONDS:
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
                },
            )

    def _get_from_cache(self, handler_type: str) -> ModelBindingConfig | None:
        """Get config from cache if present and not expired.

        Args:
            handler_type: The handler type to look up.

        Returns:
            ModelBindingConfig if cached and valid, None otherwise.
        """
        if not self._config.enable_caching:
            return None

        cached = self._cache.get(handler_type)
        if cached is None:
            return None

        if cached.is_expired():
            del self._cache[handler_type]
            self._expired_evictions += 1
            return None

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
        """
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
        inline_config: dict[str, object] | None,
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
                sources.append(f"{scheme}://...")
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
        inline_config: dict[str, object] | None,
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
        merged_config = self._apply_env_overrides(merged_config, handler_type)

        # Resolve any vault:// references in the config
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
        merged_config = self._apply_env_overrides(merged_config, handler_type)

        # Resolve any vault:// references in the config (async)
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
            config_ref: Configuration reference (file://, env:, vault:).
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
            raise ProtocolConfigurationError(
                f"Invalid config reference: {parse_result.error_message}",
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
            config_ref: Configuration reference (file://, env:, vault:).
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
            raise ProtocolConfigurationError(
                f"Invalid config reference: {parse_result.error_message}",
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

        # Resolve to absolute path for security validation
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError):
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

        # Security: Validate path is within config_dir if configured
        if self._config.config_dir is not None:
            config_dir_resolved = self._config.config_dir.resolve()
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
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Failed to retrieve configuration from Vault",
                context=context,
            ) from e

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
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="load_from_vault_async",
                target_name="binding_config_resolver",
            )
            raise ProtocolConfigurationError(
                "Failed to retrieve configuration from Vault",
                context=context,
            ) from e

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

    def _apply_env_overrides(
        self,
        config: dict[str, object],
        handler_type: str,
    ) -> dict[str, object]:
        """Apply environment variable overrides.

        Environment variables follow the pattern:
        {env_prefix}_{HANDLER_TYPE}_{FIELD}

        For example: HANDLER_DB_TIMEOUT_MS=10000

        Args:
            config: Base configuration dictionary.
            handler_type: Handler type for env var name construction.

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
                converted = self._convert_env_value(env_value, model_field)
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
    ) -> object | None:
        """Convert environment variable string to appropriate type.

        Args:
            value: String value from environment.
            field: Field name to determine type.

        Returns:
            Converted value, or None if conversion fails.
        """
        # Boolean fields
        if field == "enabled":
            return value.lower() in {"true", "1", "yes"}

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
                logger.warning(
                    "Invalid integer value in environment variable for field: %s",
                    field,
                )
                return None

        # Float fields
        if field == "rate_limit_per_second":
            try:
                return float(value)
            except ValueError:
                logger.warning(
                    "Invalid float value in environment variable for field: %s",
                    field,
                )
                return None

        # String fields
        if field in {"name", "backoff_strategy"}:
            return value

        return value

    def _resolve_vault_refs(
        self,
        config: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Resolve any vault:// references in config values.

        Scans all string values for vault:// prefix and resolves them
        using the SecretResolver.

        Args:
            config: Configuration dictionary.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Configuration with vault references resolved.
        """
        secret_resolver = self._get_secret_resolver()
        if secret_resolver is None:
            return config

        result: dict[str, object] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("vault:"):
                # Parse and resolve vault reference
                vault_path = value[6:]  # Remove "vault:" prefix
                fragment = None
                if "#" in vault_path:
                    vault_path, fragment = vault_path.rsplit("#", 1)

                logical_name = vault_path
                if fragment:
                    logical_name = f"{vault_path}#{fragment}"

                try:
                    secret = secret_resolver.get_secret(logical_name, required=False)
                    if secret is not None:
                        result[key] = secret.get_secret_value()
                    else:
                        result[key] = value  # Keep original if not found
                except Exception as e:
                    logger.warning(
                        "Failed to resolve vault reference",
                        extra={
                            "correlation_id": str(correlation_id),
                            "vault_path": logical_name,
                            "config_key": key,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    result[key] = value  # Keep original on error
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                result[key] = self._resolve_vault_refs(value, correlation_id)
            else:
                result[key] = value

        return result

    async def _resolve_vault_refs_async(
        self,
        config: dict[str, object],
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Resolve any vault:// references in config values asynchronously.

        Args:
            config: Configuration dictionary.
            correlation_id: Correlation ID for error tracking.

        Returns:
            Configuration with vault references resolved.
        """
        secret_resolver = self._get_secret_resolver()
        if secret_resolver is None:
            return config

        result: dict[str, object] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("vault:"):
                # Parse and resolve vault reference
                vault_path = value[6:]  # Remove "vault:" prefix
                fragment = None
                if "#" in vault_path:
                    vault_path, fragment = vault_path.rsplit("#", 1)

                logical_name = vault_path
                if fragment:
                    logical_name = f"{vault_path}#{fragment}"

                try:
                    secret = await secret_resolver.get_secret_async(
                        logical_name, required=False
                    )
                    if secret is not None:
                        result[key] = secret.get_secret_value()
                    else:
                        result[key] = value  # Keep original if not found
                except Exception as e:
                    logger.warning(
                        "Failed to resolve vault reference",
                        extra={
                            "correlation_id": str(correlation_id),
                            "vault_path": logical_name,
                            "config_key": key,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    result[key] = value  # Keep original on error
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                result[key] = await self._resolve_vault_refs_async(
                    value, correlation_id
                )
            else:
                result[key] = value

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
            except Exception as e:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_config",
                    target_name=f"handler:{handler_type}",
                )
                raise ProtocolConfigurationError(
                    f"Invalid retry policy configuration: {e}",
                    context=context,
                ) from e

        # Filter to only known fields if strict validation is disabled
        if not self._config.strict_validation:
            known_fields = set(ModelBindingConfig.model_fields.keys())
            config = {k: v for k, v in config.items() if k in known_fields}

        try:
            return ModelBindingConfig.model_validate(config)
        except Exception as e:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_config",
                target_name=f"handler:{handler_type}",
            )
            raise ProtocolConfigurationError(
                f"Invalid handler configuration: {e}",
                context=context,
            ) from e


__all__: Final[list[str]] = ["BindingConfigResolver"]
