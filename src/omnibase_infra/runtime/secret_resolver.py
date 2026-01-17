# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Centralized secret resolution for ONEX infrastructure.

SecretResolver provides a unified interface for accessing secrets from multiple sources:
- Vault (if configured) - **STUB: Not yet implemented, raises NotImplementedError**
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
    - File paths are never logged (prevents information disclosure)
    - Path traversal attacks are blocked for file-based secrets
    - Bootstrap secrets bypass normal resolution to prevent circular dependencies

Memory Handling:
    Raw secret values (plain strings) are briefly held in local variables during
    resolution before being wrapped in SecretStr. Python's garbage collector will
    reclaim this memory, but there is no explicit secure memory wiping. This is
    acceptable for most use cases, but for high-security environments:

    - Consider using dedicated secret management libraries with secure memory handling
    - Use short-lived processes for secret-intensive operations
    - Ensure swap is encrypted at the OS level

    The brief exposure window is minimized by immediately wrapping values in SecretStr
    after retrieval and never storing raw strings in instance attributes.

Vault Integration Status:
    The Vault integration is currently a **stub implementation**. When a secret is
    configured with source_type="vault":

    - If vault_handler is None: Returns None with a warning log (graceful degradation)
    - If vault_handler is provided: Raises NotImplementedError with guidance

    To use Vault secrets today, configure them via 'env' or 'file' source types
    until full Vault integration is implemented. See docs/patterns/secret_resolver.md
    for migration guidance.

    TODO: Implement full Vault integration (follow-up ticket required)
    Implementation will require:
    1. Create envelope for vault.read_secret operation with correlation_id
    2. Call HandlerVault.execute() for KV v2 secret retrieval
    3. Parse path to extract mount/path/field from "secret/data/path#field" format
    4. Handle error responses:
       - 404 (NotFound): Return None
       - 403 (Forbidden): Raise InfraAuthenticationError
       - Timeout: Raise InfraTimeoutError with ModelTimeoutErrorContext
       - Other: Raise SecretResolutionError with sanitized message
    5. Extract data[field] from response or first value if no field specified
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, SecretResolutionError
from omnibase_infra.runtime.models.model_cached_secret import ModelCachedSecret
from omnibase_infra.runtime.models.model_secret_cache_stats import ModelSecretCacheStats
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_info import ModelSecretSourceInfo
from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
    SecretSourceType,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers.handler_vault import HandlerVault

logger = logging.getLogger(__name__)

# Maximum file size for secret files (1MB)
# Prevents memory exhaustion from accidentally pointing at large files
MAX_SECRET_FILE_SIZE = 1024 * 1024


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
        This class supports concurrent access from both sync and async contexts
        using a two-level locking strategy:

        1. ``threading.Lock`` (``_lock``): Protects all cache reads/writes and
           stats updates. This lock is held briefly for in-memory operations.

        2. Per-key ``asyncio.Lock`` (``_async_key_locks``): Prevents duplicate
           async fetches for the SAME secret. When multiple async callers request
           the same secret simultaneously, only one performs the fetch while
           others wait and reuse the cached result. Different secrets can be
           fetched in parallel.

        Sync/Async Coordination:
            - Sync ``get_secret``: Holds ``_lock`` for entire operation (cache
              check through cache write). This ensures atomicity but may briefly
              block async callers during cache access.
            - Async ``get_secret_async``: Uses per-key async locks to serialize
              fetches for the same key, with ``_lock`` held only briefly for
              cache access. This allows parallel fetches for different secrets.

        Edge Case - Sync/Async Race:
            Due to the different locking granularity between sync (holds lock
            during I/O) and async (releases lock during I/O), there's a small
            window where both sync and async code might resolve the same secret
            simultaneously. This is handled by a check-before-write pattern:
            before caching, we verify the key isn't already present. If a sync
            caller won the race, we skip the redundant cache write.

            Why this race is acceptable:
            1. Both sync and async resolvers fetch the same value from the same
               source (env var, file, or Vault path), so the resolved values are
               always identical.
            2. The skip-if-present check prevents wasted cache writes, but even
               without it, last-write-wins produces correct results since all
               writers have the same value.
            3. There is no correctness issue - only a minor inefficiency of
               potentially resolving the same secret twice in rare cases.

        Bootstrap Secret Isolation:
            Bootstrap secrets (vault.token, vault.addr, vault.ca_cert) are
            resolved exclusively from environment variables, never from Vault
            or files. This prevents circular dependencies during Vault init.
            The resolution path is isolated from regular secrets, and cache
            writes are always protected by ``_lock`` in both sync and async
            contexts.

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
        self._hit_counts: defaultdict[str, int] = defaultdict(int)  # per logical_name
        self._lock = threading.Lock()
        # Per-key async locks to allow parallel fetches for different secrets
        # while preventing duplicate fetches for the same secret
        self._async_key_locks: dict[str, asyncio.Lock] = {}

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

        Note:
            This sync method resolves secrets sequentially. For better latency
            when resolving multiple secrets that involve I/O (Vault, file-based),
            prefer using ``get_secrets_async()`` which resolves in parallel via
            ``asyncio.gather()``.

            TODO: Consider parallelizing sync resolution using ThreadPoolExecutor
            for Vault/file secrets. This would require careful lock management to
            avoid deadlocks with the per-key locking strategy. See OMN-1374.
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
            with sync callers. Per-key async locks serialize resolution for the
            same secret while allowing parallel fetches for different secrets.

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

        # Get or create per-key async lock for this logical_name
        # This allows parallel fetches for different secrets while preventing
        # duplicate fetches for the same secret
        key_lock = self._get_async_key_lock(logical_name)

        async with key_lock:
            # Double-check cache after acquiring async lock - another coroutine may
            # have resolved this secret while we were waiting on the lock
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

    def _get_async_key_lock(self, logical_name: str) -> asyncio.Lock:
        """Get or create an async lock for a specific logical_name.

        This enables parallel resolution of different secrets while preventing
        duplicate concurrent fetches for the same secret.

        Thread Safety:
            Uses threading.Lock to safely access the key locks dictionary,
            ensuring thread-safe creation of new locks.

        Args:
            logical_name: The secret key to get a lock for

        Returns:
            asyncio.Lock for the given logical_name
        """
        with self._lock:
            if logical_name not in self._async_key_locks:
                self._async_key_locks[logical_name] = asyncio.Lock()
            return self._async_key_locks[logical_name]

    async def get_secrets_async(
        self,
        logical_names: list[str],
        required: bool = True,
    ) -> dict[str, SecretStr | None]:
        """Resolve multiple secrets asynchronously in parallel.

        Uses asyncio.gather() to fetch multiple secrets concurrently, improving
        performance when resolving multiple secrets that may involve I/O (e.g.,
        Vault or file-based secrets).

        Thread Safety:
            Each secret resolution uses per-key async locks, so fetches for
            different secrets proceed in parallel while fetches for the same
            secret are serialized.

        Args:
            logical_names: List of dotted paths
            required: If True, raises on first missing secret

        Returns:
            Dict mapping logical_name -> SecretStr | None

        Raises:
            SecretResolutionError: If required=True and any secret is not found.
                The first missing secret will raise; other fetches may complete
                or be cancelled depending on timing.
        """
        if not logical_names:
            return {}

        # Create tasks for parallel resolution
        tasks = [
            self.get_secret_async(name, required=required) for name in logical_names
        ]

        # Gather results - raises on first failure if required=True
        values = await asyncio.gather(*tasks)

        return dict(zip(logical_names, values, strict=True))

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
            self._hit_counts.pop(logical_name, None)
            self._expired_evictions += 1
            return None

        # Track hits using internal counter (model is frozen)
        self._hit_counts[logical_name] += 1
        self._hits += 1
        return cached.value

    def _is_bootstrap_secret(self, logical_name: str) -> bool:
        """Check if a logical name is a bootstrap secret.

        Bootstrap secrets are resolved ONLY from environment variables, never from
        Vault or files. This ensures they're available before Vault is initialized.

        Security:
            Bootstrap secrets (vault.token, vault.addr, vault.ca_cert) are needed
            to initialize the Vault connection. They MUST come from env vars to
            avoid a circular dependency.

        Args:
            logical_name: The logical name to check

        Returns:
            True if this is a bootstrap secret that bypasses normal resolution
        """
        return logical_name in self._config.bootstrap_secrets

    def _resolve_bootstrap_secret_value(self, logical_name: str) -> SecretStr | None:
        """Resolve a bootstrap secret value from environment variables.

        Thread Safety:
            This method only reads from environment variables (atomic on most platforms)
            and does NOT write to cache. The caller is responsible for cache writes
            with proper locking.

        Security:
            Bootstrap secrets are isolated from the normal resolution chain.
            They are ALWAYS resolved from environment variables using the
            convention-based naming (logical_name -> ENV_VAR).

        Args:
            logical_name: The bootstrap secret's logical name

        Returns:
            SecretStr if found, None if env var is not set
        """
        # Convert logical name to env var name
        env_var = self._logical_name_to_env_var(logical_name)
        value = os.environ.get(env_var)

        if value is None:
            return None

        return SecretStr(value)

    def _resolve_secret(self, logical_name: str) -> SecretStr | None:
        """Resolve secret from source and cache it.

        Thread Safety:
            This method MUST be called while holding _lock. It writes to cache
            directly without additional locking.

        Security:
            Bootstrap secrets (vault.token, vault.addr, etc.) are resolved directly
            from environment variables, bypassing the normal source chain. This
            prevents circular dependencies when initializing Vault.

        Args:
            logical_name: The logical name to resolve

        Returns:
            SecretStr if found, None otherwise
        """
        # SECURITY: Bootstrap secrets bypass normal resolution
        # They must come from env vars to avoid circular dependency with Vault
        if self._is_bootstrap_secret(logical_name):
            secret = self._resolve_bootstrap_secret_value(logical_name)
            if secret is not None:
                # Cache write is safe here - caller holds _lock
                self._cache_secret(logical_name, secret, "env")
            return secret

        source = self._get_source_spec(logical_name)
        if source is None:
            return None

        value: str | None = None

        if source.source_type == "env":
            value = os.environ.get(source.source_path)
        elif source.source_type == "file":
            value = self._read_file_secret(source.source_path, logical_name)
        elif source.source_type == "vault":
            if self._vault_handler is None:
                logger.warning(
                    "Vault handler not configured for secret: %s",
                    logical_name,
                    extra={"logical_name": logical_name},
                )
                return None
            value = self._read_vault_secret_sync(source.source_path, logical_name)

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
            Bootstrap secrets also use _lock for their cache writes to ensure
            thread-safe access from both sync and async contexts.

        Security:
            Bootstrap secrets (vault.token, vault.addr, etc.) are resolved directly
            from environment variables, bypassing the normal source chain. This
            prevents circular dependencies when initializing Vault.

        Args:
            logical_name: The logical name to resolve

        Returns:
            SecretStr if found, None otherwise
        """
        # SECURITY: Bootstrap secrets bypass normal resolution
        # They must come from env vars to avoid circular dependency with Vault
        if self._is_bootstrap_secret(logical_name):
            # Resolve value (no cache write in _resolve_bootstrap_secret_value)
            secret = self._resolve_bootstrap_secret_value(logical_name)
            if secret is not None:
                # THREAD SAFETY: Use lock for cache write to prevent race with sync callers
                # Check-before-write pattern avoids unnecessary overwrites if sync caller
                # already cached this secret between our cache check and now
                with self._lock:
                    if logical_name not in self._cache:
                        self._cache_secret(logical_name, secret, "env")
            return secret

        source = self._get_source_spec(logical_name)
        if source is None:
            return None

        value: str | None = None

        # I/O operations - NOT under lock to avoid blocking
        if source.source_type == "env":
            value = os.environ.get(source.source_path)
        elif source.source_type == "file":
            value = await asyncio.to_thread(
                self._read_file_secret, source.source_path, logical_name
            )
        elif source.source_type == "vault":
            if self._vault_handler is None:
                logger.warning(
                    "Vault handler not configured for secret: %s",
                    logical_name,
                    extra={"logical_name": logical_name},
                )
                return None
            value = await self._read_vault_secret_async(
                source.source_path, logical_name
            )

        if value is None:
            return None

        secret = SecretStr(value)
        # THREAD SAFETY: Use lock for cache write to prevent race with sync callers
        # Check-before-write pattern avoids unnecessary overwrites if sync caller
        # already cached this secret between our cache check and now
        with self._lock:
            if logical_name not in self._cache:
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

    def _read_file_secret(self, path: str, logical_name: str = "") -> str | None:
        """Read secret from file.

        Thread Safety:
            This method avoids TOCTOU race conditions by catching exceptions
            during the read operation rather than pre-checking file existence.

        Security:
            - Path traversal attacks are prevented by validating resolved paths
              stay within the configured secrets_dir
            - Error messages are sanitized to avoid leaking path information
            - No secret values are ever logged

        Args:
            path: Path to the secret file (absolute or relative to secrets_dir)
            logical_name: The logical name being resolved (for error context only)

        Returns:
            Secret value with whitespace stripped, or None if not found or unreadable
        """
        secret_path = Path(path)

        # Track whether the original path was relative BEFORE combining with secrets_dir
        # This is critical for path traversal detection
        original_is_relative = not secret_path.is_absolute()

        # If relative path, resolve against secrets_dir
        if original_is_relative:
            secret_path = self._config.secrets_dir / path

        # Resolve to absolute path to detect path traversal
        try:
            resolved_path = secret_path.resolve()
        except (OSError, RuntimeError):
            # resolve() can fail on invalid paths or symlink loops
            logger.warning(
                "Invalid secret path for logical name: %s",
                logical_name,
                extra={"logical_name": logical_name},
            )
            return None

        # SECURITY: Prevent path traversal attacks
        # Verify the resolved path is within secrets_dir for relative paths
        # Absolute paths are trusted (explicitly configured by administrator)
        if original_is_relative:
            secrets_dir_resolved = self._config.secrets_dir.resolve()
            # Relative paths MUST resolve within secrets_dir
            try:
                resolved_path.relative_to(secrets_dir_resolved)
            except ValueError:
                # Path escapes secrets_dir - this is a path traversal attempt
                logger.warning(
                    "Path traversal detected for secret: %s",
                    logical_name,
                    extra={"logical_name": logical_name},
                )
                return None

        # Avoid TOCTOU race: catch exceptions during read instead of pre-checking
        try:
            # Check file size to prevent memory exhaustion from large files
            file_size = resolved_path.stat().st_size
            if file_size > MAX_SECRET_FILE_SIZE:
                logger.warning(
                    "Secret file exceeds size limit (%d bytes): %s",
                    file_size,
                    logical_name,
                    extra={"logical_name": logical_name, "file_size": file_size},
                )
                return None
            return resolved_path.read_text().strip()
        except FileNotFoundError:
            # File does not exist - this is expected for optional secrets
            # SECURITY: Don't log the actual path to avoid information disclosure
            logger.debug(
                "Secret file not found for logical name: %s",
                logical_name,
                extra={"logical_name": logical_name},
            )
            return None
        except IsADirectoryError:
            # Path exists but is a directory, not a file
            # SECURITY: Don't log the actual path
            logger.warning(
                "Secret path is a directory for logical name: %s",
                logical_name,
                extra={"logical_name": logical_name},
            )
            return None
        except PermissionError:
            # Permission denied - log at warning level since this may indicate
            # a configuration issue (file exists but is not readable)
            # SECURITY: Don't log the actual path
            logger.warning(
                "Permission denied reading secret for logical name: %s",
                logical_name,
                extra={"logical_name": logical_name},
            )
            return None
        except OSError as e:
            # Catch other OS-level errors (e.g., too many open files, I/O errors)
            # SECURITY: Don't log the path or detailed OS error which may leak info
            logger.warning(
                "OS error reading secret for logical name: %s (error type: %s)",
                logical_name,
                type(e).__name__,
                extra={"logical_name": logical_name, "error_type": type(e).__name__},
            )
            return None

    def _read_vault_secret_sync(self, path: str, logical_name: str = "") -> str | None:
        """Read secret from Vault synchronously.

        Path format: "secret/data/path#field" or "secret/data/path" (returns first field)

        Security:
            - This method never logs Vault paths (could reveal secret structure)
            - Secret values are never logged at any level
            - Error messages are sanitized to include only logical names

        Args:
            path: Vault path with optional field specifier
            logical_name: The logical name being resolved (for error context only)

        Returns:
            Secret value or None if not found

        Raises:
            NotImplementedError: Vault integration is not yet implemented.
                When implemented, this method will:
                1. Use HandlerVault.execute() with a sync wrapper
                2. Parse the path to extract mount/path/field
                3. Return the secret value or None if not found
                4. Raise SecretResolutionError on Vault communication failures

        Note:
            Vault integration requires OMN-1374 (follow-up ticket).
            Currently, configure secrets via 'env' or 'file' source types.
        """
        if self._vault_handler is None:
            return None

        # Parse path and field (validated but not logged for security)
        vault_path, field = self._parse_vault_path(path)
        # Suppress unused variable warnings - these will be used when implemented
        _ = vault_path, field

        # SECURITY: Do not log Vault paths as they reveal secret structure
        raise NotImplementedError(
            f"Vault secret resolution not yet implemented for logical name: "
            f"{logical_name}. Configure this secret via 'env' or 'file' source "
            f"until Vault integration is complete. "
            f"See docs/patterns/secret_resolver.md for migration guidance."
        )

    async def _read_vault_secret_async(
        self, path: str, logical_name: str = ""
    ) -> str | None:
        """Read secret from Vault asynchronously.

        Path format: "secret/data/path#field" or "secret/data/path" (returns first field)

        Security:
            - This method never logs Vault paths (could reveal secret structure)
            - Secret values are never logged at any level
            - Error messages are sanitized to include only logical names

        Args:
            path: Vault path with optional field specifier
            logical_name: The logical name being resolved (for error context only)

        Returns:
            Secret value or None if not found

        Raises:
            NotImplementedError: Vault integration is not yet implemented.
                When implemented, this method will:
                1. Use HandlerVault.execute() for async Vault communication
                2. Parse the path to extract mount/path/field
                3. Return the secret value or None if not found
                4. Raise SecretResolutionError on Vault communication failures

        Implementation Plan (for follow-up ticket):
            1. Create envelope for vault.read_secret operation
            2. Call self._vault_handler.execute(envelope)
            3. Handle response:
               - Success: Extract data[field] or first value
               - NotFound: Return None
               - Auth failure: Raise InfraAuthenticationError
               - Timeout: Raise InfraTimeoutError
               - Other: Raise SecretResolutionError

        Note:
            Vault integration requires OMN-1374 (follow-up ticket).
            Currently, configure secrets via 'env' or 'file' source types.
        """
        if self._vault_handler is None:
            return None

        # Parse path and field (validated but not logged for security)
        vault_path, field = self._parse_vault_path(path)
        # Suppress unused variable warnings - these will be used when implemented
        _ = vault_path, field

        # SECURITY: Do not log Vault paths as they reveal secret structure
        raise NotImplementedError(
            f"Vault secret resolution not yet implemented for logical name: "
            f"{logical_name}. Configure this secret via 'env' or 'file' source "
            f"until Vault integration is complete. "
            f"See docs/patterns/secret_resolver.md for migration guidance."
        )

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
        source_type: SecretSourceType,
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

    def _get_ttl(self, logical_name: str, source_type: SecretSourceType) -> int:
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


__all__: list[str] = ["SecretResolver", "SecretSourceType"]
