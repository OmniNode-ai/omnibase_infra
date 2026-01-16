# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Centralized secret resolution for ONEX infrastructure.

SecretResolver provides a unified interface for accessing secrets from multiple sources:
- Vault (if configured)
- Environment variables
- File-based secrets (K8s /run/secrets)

Design Philosophy:
- Dumb and deterministic: resolves and caches, does not discover or mutate
- Explicit mappings preferred, convention fallback optional
- Bootstrap secrets (Vault token/addr) always from env

Example:
    Bootstrap phase (env-only for Vault credentials)::

        vault_token = os.environ.get("VAULT_TOKEN")
        vault_addr = os.environ.get("VAULT_ADDR")

    Initialize resolver::

        config = ModelSecretResolverConfig(mappings=[...])
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

    Resolve secrets::

        db_password = resolver.get_secret("database.postgres.password")
        api_key = resolver.get_secret("llm.openai.api_key", required=False)

Security Considerations:
    - Secret values are wrapped in SecretStr to prevent accidental logging
    - Cache stores SecretStr values, never raw strings
    - Introspection methods never expose secret values
    - Error messages are sanitized to exclude secret values
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import SecretStr

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, SecretResolutionError
from omnibase_infra.runtime.models.model_cached_secret import ModelCachedSecret
from omnibase_infra.runtime.models.model_secret_cache_stats import ModelSecretCacheStats
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_info import ModelSecretSourceInfo
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec

if TYPE_CHECKING:
    from omnibase_infra.handlers.handler_vault import HandlerVault

logger = logging.getLogger(__name__)

# Type alias for secret source types
SourceType = Literal["env", "vault", "file"]


class SecretResolver:
    """Centralized secret resolution. Dumb and deterministic.

    The SecretResolver provides a unified interface for accessing secrets from
    multiple sources with caching and optional convention-based fallback.

    Resolution Order:
        1. Check cache (if not expired)
        2. Try explicit mapping from configuration
        3. Try convention fallback (if enabled): logical_name -> ENV_VAR
        4. Raise or return None based on required flag

    Thread Safety:
        - All cache operations use threading.Lock for thread-safe access
        - Async operations additionally use asyncio.Lock to serialize resolution
        - The threading lock protects cache reads/writes (fast, in-memory)
        - The async lock prevents duplicate fetches for the same secret

    Example:
        >>> config = ModelSecretResolverConfig(
        ...     mappings=[
        ...         ModelSecretMapping(
        ...             logical_name="database.postgres.password",
        ...             source=ModelSecretSourceSpec(
        ...                 source_type="env",
        ...                 source_path="POSTGRES_PASSWORD"
        ...             )
        ...         )
        ...     ]
        ... )
        >>> resolver = SecretResolver(config=config)
        >>> password = resolver.get_secret("database.postgres.password")
    """

    def __init__(
        self,
        config: ModelSecretResolverConfig,
        vault_handler: HandlerVault | None = None,
    ) -> None:
        """Initialize SecretResolver.

        Args:
            config: Resolver configuration with mappings and TTLs
            vault_handler: Optional Vault handler for Vault-sourced secrets
        """
        self._config = config
        self._vault_handler = vault_handler
        self._cache: dict[str, ModelCachedSecret] = {}
        # Track mutable stats internally since ModelSecretCacheStats is frozen
        self._hits = 0
        self._misses = 0
        self._expired_evictions = 0
        self._refreshes = 0
        self._hit_counts: dict[str, int] = {}  # Track hit counts per logical_name
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None

        # Build lookup table from mappings
        self._mappings: dict[str, ModelSecretSourceSpec] = {
            m.logical_name: m.source for m in config.mappings
        }
        self._ttl_overrides: dict[str, int] = {
            m.logical_name: m.ttl_seconds
            for m in config.mappings
            if m.ttl_seconds is not None
        }

    # === Primary API (Sync) ===

    def get_secret(
        self,
        logical_name: str,
        required: bool = True,
    ) -> SecretStr | None:
        """Resolve a secret by logical name.

        Resolution order:
            1. Check cache (if not expired)
            2. Try explicit mapping
            3. Try convention fallback (if enabled)
            4. Raise or return None based on required flag

        Args:
            logical_name: Dotted path (e.g., "database.postgres.password")
            required: If True, raises SecretResolutionError when not found

        Returns:
            SecretStr if found, None if not found and required=False

        Raises:
            SecretResolutionError: If required=True and secret not found
        """
        with self._lock:
            # Check cache first
            cached = self._get_from_cache(logical_name)
            if cached is not None:
                return cached

            # Resolve from source
            result = self._resolve_secret(logical_name)

            if result is None:
                self._misses += 1
                if required:
                    context = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="get_secret",
                        target_name="secret_resolver",
                    )
                    raise SecretResolutionError(
                        f"Secret not found: {logical_name}",
                        context=context,
                        logical_name=logical_name,
                    )
                return None

            return result

    def get_secrets(
        self,
        logical_names: list[str],
        required: bool = True,
    ) -> dict[str, SecretStr | None]:
        """Resolve multiple secrets.

        Args:
            logical_names: List of dotted paths
            required: If True, raises on first missing secret

        Returns:
            Dict mapping logical_name -> SecretStr | None
        """
        return {
            name: self.get_secret(name, required=required) for name in logical_names
        }

    # === Primary API (Async) ===

    async def get_secret_async(
        self,
        logical_name: str,
        required: bool = True,
    ) -> SecretStr | None:
        """Async wrapper for get_secret.

        For Vault secrets, this uses async I/O. For env/file secrets,
        this wraps the sync call in a thread executor.

        Thread Safety:
            Uses threading.Lock for cache access to prevent race conditions
            with sync callers. The async lock serializes resolution to prevent
            duplicate fetches.

        Args:
            logical_name: Dotted path (e.g., "database.postgres.password")
            required: If True, raises SecretResolutionError when not found

        Returns:
            SecretStr if found, None if not found and required=False

        Raises:
            SecretResolutionError: If required=True and secret not found
        """
        # Use threading lock for cache check (fast operation, prevents race with sync)
        with self._lock:
            cached = self._get_from_cache(logical_name)
            if cached is not None:
                return cached

        # Use async lock to serialize resolution (prevents duplicate fetches)
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            # Double-check cache after acquiring async lock
            # (another coroutine may have resolved while we waited)
            with self._lock:
                cached = self._get_from_cache(logical_name)
                if cached is not None:
                    return cached

            # Resolve from source (potentially async for Vault)
            # Note: _resolve_secret_async handles its own locking for cache writes
            result = await self._resolve_secret_async(logical_name)

            if result is None:
                with self._lock:
                    self._misses += 1
                if required:
                    context = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="get_secret_async",
                        target_name="secret_resolver",
                    )
                    raise SecretResolutionError(
                        f"Secret not found: {logical_name}",
                        context=context,
                        logical_name=logical_name,
                    )
                return None

            return result

    async def get_secrets_async(
        self,
        logical_names: list[str],
        required: bool = True,
    ) -> dict[str, SecretStr | None]:
        """Resolve multiple secrets asynchronously.

        Args:
            logical_names: List of dotted paths
            required: If True, raises on first missing secret

        Returns:
            Dict mapping logical_name -> SecretStr | None
        """
        results: dict[str, SecretStr | None] = {}
        for name in logical_names:
            results[name] = await self.get_secret_async(name, required=required)
        return results

    # === Cache Management ===

    def refresh(self, logical_name: str) -> None:
        """Force refresh a single secret (invalidate cache).

        Args:
            logical_name: The logical name to refresh
        """
        with self._lock:
            if logical_name in self._cache:
                del self._cache[logical_name]
                if logical_name in self._hit_counts:
                    del self._hit_counts[logical_name]
                self._refreshes += 1

    def refresh_all(self) -> None:
        """Force refresh all cached secrets."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hit_counts.clear()
            self._refreshes += count

    def get_cache_stats(self) -> ModelSecretCacheStats:
        """Return cache statistics.

        Returns:
            ModelSecretCacheStats with hit/miss/refresh counts
        """
        with self._lock:
            return ModelSecretCacheStats(
                total_entries=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                refreshes=self._refreshes,
                expired_evictions=self._expired_evictions,
            )

    # === Introspection (non-sensitive) ===

    def list_configured_secrets(self) -> list[str]:
        """List all configured logical names (not values).

        Returns:
            List of logical names from configuration
        """
        return list(self._mappings.keys())

    def get_source_info(self, logical_name: str) -> ModelSecretSourceInfo | None:
        """Return source type and masked path for a logical name.

        This method is safe to use for debugging and monitoring as it
        never exposes actual secret values.

        Args:
            logical_name: The logical name to inspect

        Returns:
            ModelSecretSourceInfo with masked path, or None if not configured
        """
        source = self._get_source_spec(logical_name)
        if source is None:
            return None

        # Mask sensitive parts of the path
        masked_path = self._mask_source_path(source)

        # Use lock for thread-safe cache access
        with self._lock:
            cached_entry = self._cache.get(logical_name)
            return ModelSecretSourceInfo(
                logical_name=logical_name,
                source_type=source.source_type,
                source_path_masked=masked_path,
                is_cached=cached_entry is not None,
                expires_at=cached_entry.expires_at if cached_entry else None,
            )

    # === Internal Methods ===

    def _get_from_cache(self, logical_name: str) -> SecretStr | None:
        """Get secret from cache if present and not expired.

        Args:
            logical_name: The logical name to look up

        Returns:
            SecretStr if cached and valid, None otherwise
        """
        cached = self._cache.get(logical_name)
        if cached is None:
            return None

        if cached.is_expired():
            del self._cache[logical_name]
            if logical_name in self._hit_counts:
                del self._hit_counts[logical_name]
            self._expired_evictions += 1
            return None

        # Track hits using internal counter (model is frozen)
        self._hit_counts[logical_name] = self._hit_counts.get(logical_name, 0) + 1
        self._hits += 1
        return cached.value

    def _resolve_secret(self, logical_name: str) -> SecretStr | None:
        """Resolve secret from source and cache it.

        Args:
            logical_name: The logical name to resolve

        Returns:
            SecretStr if found, None otherwise
        """
        source = self._get_source_spec(logical_name)
        if source is None:
            return None

        value: str | None = None

        if source.source_type == "env":
            value = os.environ.get(source.source_path)
        elif source.source_type == "file":
            value = self._read_file_secret(source.source_path)
        elif source.source_type == "vault":
            if self._vault_handler is None:
                logger.warning(
                    "Vault handler not configured for secret: %s",
                    logical_name,
                    extra={"logical_name": logical_name},
                )
                return None
            value = self._read_vault_secret_sync(source.source_path)

        if value is None:
            return None

        secret = SecretStr(value)
        self._cache_secret(logical_name, secret, source.source_type)
        return secret

    async def _resolve_secret_async(self, logical_name: str) -> SecretStr | None:
        """Resolve secret from source asynchronously.

        Thread Safety:
            Uses threading.Lock for cache writes to prevent race conditions
            with sync callers. I/O operations are performed outside the lock.

        Args:
            logical_name: The logical name to resolve

        Returns:
            SecretStr if found, None otherwise
        """
        source = self._get_source_spec(logical_name)
        if source is None:
            return None

        value: str | None = None

        # I/O operations - NOT under lock to avoid blocking
        if source.source_type == "env":
            value = os.environ.get(source.source_path)
        elif source.source_type == "file":
            value = await asyncio.to_thread(self._read_file_secret, source.source_path)
        elif source.source_type == "vault":
            if self._vault_handler is None:
                logger.warning(
                    "Vault handler not configured for secret: %s",
                    logical_name,
                    extra={"logical_name": logical_name},
                )
                return None
            value = await self._read_vault_secret_async(source.source_path)

        if value is None:
            return None

        secret = SecretStr(value)
        # Use threading lock for cache write (fast operation, prevents race with sync)
        with self._lock:
            self._cache_secret(logical_name, secret, source.source_type)
        return secret

    def _get_source_spec(self, logical_name: str) -> ModelSecretSourceSpec | None:
        """Get source spec from mapping or convention fallback.

        Args:
            logical_name: The logical name to look up

        Returns:
            ModelSecretSourceSpec if found, None otherwise
        """
        # Try explicit mapping first
        if logical_name in self._mappings:
            return self._mappings[logical_name]

        # Try convention fallback
        if self._config.enable_convention_fallback:
            env_var = self._logical_name_to_env_var(logical_name)
            return ModelSecretSourceSpec(source_type="env", source_path=env_var)

        return None

    def _logical_name_to_env_var(self, logical_name: str) -> str:
        """Convert dotted logical name to environment variable name.

        Example:
            "database.postgres.password" -> "DATABASE_POSTGRES_PASSWORD"
            With prefix "ONEX_": "database.postgres.password" -> "ONEX_DATABASE_POSTGRES_PASSWORD"

        Args:
            logical_name: Dotted path to convert

        Returns:
            Environment variable name
        """
        env_var = logical_name.upper().replace(".", "_")
        if self._config.convention_env_prefix:
            env_var = f"{self._config.convention_env_prefix}{env_var}"
        return env_var

    def _read_file_secret(self, path: str) -> str | None:
        """Read secret from file.

        Args:
            path: Path to the secret file (absolute or relative to secrets_dir)

        Returns:
            Secret value with whitespace stripped, or None if not found or unreadable
        """
        secret_path = Path(path)

        # If relative path, resolve against secrets_dir
        if not secret_path.is_absolute():
            secret_path = self._config.secrets_dir / path

        if not secret_path.exists() or not secret_path.is_file():
            return None

        try:
            return secret_path.read_text().strip()
        except PermissionError:
            # Treat permission errors as "not found" - the secret is not accessible
            logger.warning(
                "Permission denied reading secret file: %s",
                secret_path,
                extra={"path": str(secret_path)},
            )
            return None

    def _read_vault_secret_sync(self, path: str) -> str | None:
        """Read secret from Vault synchronously.

        Path format: "secret/data/path#field" or "secret/data/path" (returns first field)

        Args:
            path: Vault path with optional field specifier

        Returns:
            Secret value or None if not found
        """
        if self._vault_handler is None:
            return None

        # Parse path and field
        vault_path, field = self._parse_vault_path(path)

        # TODO: Integrate with HandlerVault.execute() method
        # This will require running the async handler in a sync context
        # For now, return None - Vault integration will be added in follow-up
        logger.debug(
            "Vault secret resolution not yet implemented: %s",
            vault_path,
            extra={"vault_path": vault_path, "field": field},
        )
        return None

    async def _read_vault_secret_async(self, path: str) -> str | None:
        """Read secret from Vault asynchronously.

        Path format: "secret/data/path#field" or "secret/data/path" (returns first field)

        Args:
            path: Vault path with optional field specifier

        Returns:
            Secret value or None if not found
        """
        if self._vault_handler is None:
            return None

        # Parse path and field
        vault_path, field = self._parse_vault_path(path)

        # TODO: Integrate with HandlerVault.execute() method
        # Example envelope:
        # {
        #     "operation": "vault.read_secret",
        #     "payload": {"path": vault_path},
        #     "correlation_id": str(uuid4()),
        # }
        # result = await self._vault_handler.execute(envelope)
        # if result.status == "success" and result.payload:
        #     data = result.payload.get("data", {})
        #     if field:
        #         return data.get(field)
        #     else:
        #         # Return first value if no field specified
        #         return next(iter(data.values()), None) if data else None
        logger.debug(
            "Vault secret resolution not yet implemented: %s",
            vault_path,
            extra={"vault_path": vault_path, "field": field},
        )
        return None

    def _parse_vault_path(self, path: str) -> tuple[str, str | None]:
        """Parse Vault path into path and optional field.

        Examples:
            "secret/data/db#password" -> ("secret/data/db", "password")
            "secret/data/db" -> ("secret/data/db", None)

        Args:
            path: Vault path with optional field specifier

        Returns:
            Tuple of (vault_path, field_name or None)
        """
        if "#" in path:
            vault_path, field = path.rsplit("#", 1)
            return vault_path, field
        return path, None

    def _cache_secret(
        self,
        logical_name: str,
        value: SecretStr,
        source_type: SourceType,
    ) -> None:
        """Cache a resolved secret with appropriate TTL.

        Args:
            logical_name: The logical name being cached
            value: The secret value to cache
            source_type: Source type for TTL selection
        """
        ttl_seconds = self._get_ttl(logical_name, source_type)
        now = datetime.now(UTC)

        self._cache[logical_name] = ModelCachedSecret(
            value=value,
            source_type=source_type,
            logical_name=logical_name,
            cached_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )

    def _get_ttl(self, logical_name: str, source_type: SourceType) -> int:
        """Get TTL for a secret based on source type or override.

        Args:
            logical_name: The logical name for TTL override lookup
            source_type: Source type for default TTL selection

        Returns:
            TTL in seconds
        """
        # Check for explicit override
        if logical_name in self._ttl_overrides:
            return self._ttl_overrides[logical_name]

        # Use default based on source type
        ttl_defaults = {
            "env": self._config.default_ttl_env_seconds,
            "file": self._config.default_ttl_file_seconds,
            "vault": self._config.default_ttl_vault_seconds,
        }
        return ttl_defaults.get(source_type, self._config.default_ttl_env_seconds)

    def _mask_source_path(self, source: ModelSecretSourceSpec) -> str:
        """Mask sensitive parts of source path for introspection.

        This ensures that introspection never reveals sensitive information
        while still being useful for debugging.

        Args:
            source: Source specification to mask

        Returns:
            Masked path string safe for logging/display
        """
        if source.source_type == "env":
            # Show env var name but mask the value context
            return f"env:{source.source_path}"
        elif source.source_type == "file":
            # Show directory but mask filename
            path = Path(source.source_path)
            return f"file:{path.parent}/***"
        elif source.source_type == "vault":
            # Show mount but mask the rest
            parts = source.source_path.split("/")
            if len(parts) > 2:
                return f"vault:{parts[0]}/{parts[1]}/***"
            return "vault:***"
        return "***"


__all__: list[str] = ["SecretResolver", "SourceType"]
