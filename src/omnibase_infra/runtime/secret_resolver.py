# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Centralized secret resolution for ONEX infrastructure.

SecretResolver provides a unified interface for accessing secrets from multiple sources:
- Vault (via HandlerVault for KV v2 secrets engine)
- Environment variables
- File-based secrets (K8s /run/secrets)

Design Philosophy:
- Dumb and deterministic: resolves and caches, does not discover or mutate
- Explicit mappings preferred, convention fallback optional
- Bootstrap secrets (Vault token/addr) always from env
- Vault is treated as an injected dependency, SecretResolver owns mapping + caching + policy

Example:
    Bootstrap phase (env-only for Vault credentials)::

        vault_token = os.environ.get("VAULT_TOKEN")
        vault_addr = os.environ.get("VAULT_ADDR")

    Initialize resolver with Vault handler::

        vault_handler = HandlerVault()
        await vault_handler.initialize({...})

        config = ModelSecretResolverConfig(mappings=[...])
        resolver = SecretResolver(config=config, vault_handler=vault_handler)

    Resolve secrets with correlation ID for tracing::

        db_password = resolver.get_secret(
            "database.postgres.password",
            correlation_id=request.correlation_id,
        )
        api_key = await resolver.get_secret_async(
            "llm.openai.api_key",
            required=False,
            correlation_id=request.correlation_id,
        )

    Get resolution metrics::

        metrics = resolver.get_resolution_metrics()
        # ModelSecretResolverMetrics(success_counts={"env": 5, "vault": 3}, ...)

Security Considerations:
    - Secret values are wrapped in SecretStr to prevent accidental logging
    - Cache stores SecretStr values, never raw strings
    - Introspection methods never expose secret values
    - Error messages are sanitized to exclude secret values
    - File paths are never logged (prevents information disclosure)
    - Path traversal attacks are blocked for file-based secrets
    - Bootstrap secrets bypass normal resolution to prevent circular dependencies
    - Vault paths are never logged (could reveal secret structure)

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

Vault Integration:
    Vault secrets are resolved via HandlerVault (KV v2 secrets engine only).

    Path Format: "mount_point/path/to/secret#field"
        - mount_point: The secrets engine mount (e.g., "secret")
        - path: The secret path within the mount (e.g., "myapp/db")
        - field: Optional specific field to extract (e.g., "password")

    Examples:
        - "secret/myapp/db#password" -> Reads "password" field from secret at myapp/db
        - "secret/myapp/db" -> Reads first field value from secret

    Type Handling:
        All Vault values are converted to strings. This is intentional because
        SecretResolver returns SecretStr values (which only wrap strings) for
        security. Non-string Vault values are converted via Python's str():

        - Integers: 123 -> "123"
        - Booleans: True -> "True"
        - Lists/Dicts: Python repr (NOT JSON)

        Best Practice: Store secrets as strings in Vault. For structured data,
        store as JSON strings and parse after resolution.

    Graceful Degradation:
        - If vault_handler is None: Returns None with a warning log
        - Vault errors are wrapped in SecretResolutionError with correlation ID

    Error Handling:
        - InfraAuthenticationError: Auth failures (403)
        - InfraTimeoutError: Request timeouts
        - InfraUnavailableError: Circuit breaker open
        - SecretResolutionError: Other Vault errors (sanitized message)

Observability (OMN-1374):
    SecretResolver includes built-in metrics tracking:

    - Resolution latency by source type (env, file, vault, cache)
    - Cache hit/miss rates (via get_cache_stats())
    - Resolution success/failure counts by source type

    External metrics collection via ProtocolSecretResolverMetrics:
        metrics_collector = MyPrometheusMetrics()
        resolver = SecretResolver(config=config, metrics_collector=metrics_collector)

    Structured logging includes:
        - logical_name: The secret being resolved
        - source_type: Where the secret came from
        - cache_hit: Whether it was a cache hit
        - correlation_id: For distributed tracing
        - latency_ms: Resolution time (on success)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from uuid import UUID, uuid4

from pydantic import SecretStr

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    SecretResolutionError,
)
from omnibase_infra.runtime.models.model_cached_secret import ModelCachedSecret
from omnibase_infra.runtime.models.model_secret_cache_stats import ModelSecretCacheStats
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_resolver_metrics import (
    ModelSecretResolverMetrics,
)
from omnibase_infra.runtime.models.model_secret_source_info import ModelSecretSourceInfo
from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
    SecretSourceType,
)
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.handlers.handler_vault import HandlerVault


logger = logging.getLogger(__name__)


@runtime_checkable
class ProtocolSecretResolverMetrics(Protocol):
    """Protocol for SecretResolver metrics collection.

    Implementations can hook into secret resolution operations to collect:
    - Resolution latency by source type (env, file, vault, cache)
    - Cache hit/miss rates
    - Resolution failure counts by source type
    - Success counts by source type

    All methods are optional (duck-typed). If a method is not implemented,
    the metric simply won't be recorded.

    Thread Safety:
        Implementations MUST be thread-safe because SecretResolver's
        ``_record_resolution_success`` and ``_record_resolution_failure`` methods
        may be called concurrently from multiple threads during parallel
        secret resolution operations. This includes:

        - Multiple sync callers resolving different secrets simultaneously
        - Async callers running in parallel via ``asyncio.gather``
        - Mixed sync/async access patterns during bootstrap and runtime

        Thread-Safe Primitives (recommended):
            - ``threading.Lock``: Protects counter increments and dict updates
            - ``threading.RLock``: For reentrant access (if metrics methods call each other)
            - ``collections.Counter`` with lock protection: Convenient for source_type counts
            - ``prometheus_client``: Inherently thread-safe (Counter, Histogram, Gauge)
            - ``queue.Queue``: For async metric collection (producer-consumer pattern)

    Example Implementation (threading.Lock)::

        import threading
        from collections import defaultdict

        class ThreadSafeSecretResolverMetrics:
            '''Minimal thread-safe metrics using threading.Lock.'''

            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._latencies: list[tuple[str, float]] = []
                self._cache_hits = 0
                self._cache_misses = 0
                self._success_counts: defaultdict[str, int] = defaultdict(int)
                self._failure_counts: defaultdict[str, int] = defaultdict(int)

            def record_resolution_latency(
                self, source_type: str, latency_ms: float
            ) -> None:
                with self._lock:
                    self._latencies.append((source_type, latency_ms))

            def record_cache_hit(self) -> None:
                with self._lock:
                    self._cache_hits += 1

            def record_cache_miss(self) -> None:
                with self._lock:
                    self._cache_misses += 1

            def record_resolution_success(self, source_type: str) -> None:
                with self._lock:
                    self._success_counts[source_type] += 1

            def record_resolution_failure(self, source_type: str) -> None:
                with self._lock:
                    self._failure_counts[source_type] += 1

    Example Implementation (prometheus_client)::

        from prometheus_client import Counter, Histogram

        class PrometheusSecretResolverMetrics:
            def __init__(self) -> None:
                self._latency = Histogram(
                    'secret_resolution_latency_ms',
                    'Latency of secret resolution in milliseconds',
                    ['source_type'],
                )
                self._cache_hits = Counter('secret_cache_hits_total', 'Cache hits')
                self._cache_misses = Counter('secret_cache_misses_total', 'Cache misses')
                self._failures = Counter(
                    'secret_resolution_failures_total',
                    'Resolution failures',
                    ['source_type'],
                )

            def record_resolution_latency(
                self, source_type: str, latency_ms: float
            ) -> None:
                self._latency.labels(source_type=source_type).observe(latency_ms)

            def record_cache_hit(self) -> None:
                self._cache_hits.inc()

            def record_cache_miss(self) -> None:
                self._cache_misses.inc()

            def record_resolution_failure(self, source_type: str) -> None:
                self._failures.labels(source_type=source_type).inc()
    """

    def record_resolution_latency(self, source_type: str, latency_ms: float) -> None:
        """Record latency for a secret resolution operation.

        Args:
            source_type: Source type (env, file, vault, cache)
            latency_ms: Time in milliseconds for the operation
        """
        ...

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        ...

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        ...

    def record_resolution_success(self, source_type: str) -> None:
        """Record a successful resolution.

        Args:
            source_type: Source type (env, file, vault)
        """
        ...

    def record_resolution_failure(self, source_type: str) -> None:
        """Record a resolution failure.

        Args:
            source_type: Source type (env, file, vault)
        """
        ...


# Maximum file size for secret files (1MB)
# Prevents memory exhaustion from accidentally pointing at large files
MAX_SECRET_FILE_SIZE = 1024 * 1024

# Maximum latency samples to retain for metrics (rolling window)
MAX_LATENCY_SAMPLES = 1000

# Warning threshold for async key locks dictionary size (DoS mitigation)
ASYNC_KEY_LOCKS_WARNING_THRESHOLD = 1000

# Maximum async key locks before LRU eviction (memory leak prevention)
# When this limit is reached, oldest 10% of entries are evicted
MAX_ASYNC_KEY_LOCKS = 1000

# Cache TTL jitter percentage for symmetric ±10% jitter (stampede prevention)
CACHE_TTL_JITTER_PERCENT = 0.1

# Rate limit interval for LRU eviction warnings (prevents log flooding)
EVICTION_WARNING_INTERVAL_SECONDS = 60.0


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

        1. ``threading.RLock`` (``_lock``): Protects all cache reads/writes and
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
        metrics_collector: ProtocolSecretResolverMetrics | None = None,
    ) -> None:
        """Initialize SecretResolver.

        Args:
            config: Resolver configuration with mappings and TTLs
            vault_handler: Optional Vault handler for Vault-sourced secrets
            metrics_collector: Optional external metrics collector for observability

        Note:
            For ONEX applications using ``ModelONEXContainer``, consider resolving
            dependencies via container-based DI rather than direct constructor
            injection. This enables centralized lifecycle management and consistent
            dependency resolution across the application. The current explicit
            constructor parameters are retained for flexibility in standalone usage
            and testing scenarios.
        """
        self._config = config
        self._vault_handler = vault_handler
        self._metrics_collector = metrics_collector
        self._cache: dict[str, ModelCachedSecret] = {}
        # Track mutable stats internally since ModelSecretCacheStats is frozen
        self._hits = 0
        self._misses = 0
        self._expired_evictions = 0
        self._refreshes = 0
        self._hit_counts: defaultdict[str, int] = defaultdict(int)  # per logical_name
        # RLock (reentrant lock) allows the same thread to acquire the lock
        # multiple times, which is needed because get_secret() holds the lock
        # while calling _record_resolution_success() which also needs the lock.
        self._lock = threading.RLock()
        # Per-key async locks to allow parallel fetches for different secrets
        # while preventing duplicate fetches for the same secret
        self._async_key_locks: dict[str, asyncio.Lock] = {}

        # === Metrics Tracking (OMN-1374) ===
        # Resolution latency tracking (deque of (source_type, latency_ms) tuples)
        self._resolution_latencies: deque[tuple[str, float]] = deque(
            maxlen=MAX_LATENCY_SAMPLES
        )
        # Resolution success/failure counts by source type
        self._resolution_success_counts: defaultdict[str, int] = defaultdict(int)
        self._resolution_failure_counts: defaultdict[str, int] = defaultdict(int)

        # === Rate Limiting for Warnings ===
        # Track last eviction warning time to rate-limit log output
        self._last_eviction_warning_time: float = 0.0

        # Build lookup table from mappings
        self._mappings: dict[str, ModelSecretSourceSpec] = {
            m.logical_name: m.source for m in config.mappings
        }
        self._ttl_overrides: dict[str, int] = {
            m.logical_name: m.ttl_seconds
            for m in config.mappings
            if m.ttl_seconds is not None
        }

    # === Container-Based Factory ===

    @classmethod
    async def from_container(
        cls,
        container: ModelONEXContainer,
        config: ModelSecretResolverConfig,
    ) -> SecretResolver:
        """Create SecretResolver from ONEX container with dependency injection.

        This async factory method supports the ONEX container-based dependency
        injection pattern while the regular constructor remains available for
        standalone use and testing scenarios.

        The factory attempts to resolve optional dependencies (HandlerVault,
        metrics collector) from the container's service registry. If a dependency
        is not registered, the resolver is created without it - this allows
        graceful degradation when optional services are unavailable.

        Parameters:
            This factory method accepts 2 parameters (both required):

            - ``container`` (required): The ONEX container with registered services
            - ``config`` (required): Resolver configuration with secret mappings

        Args:
            container: ONEX dependency injection container. May have HandlerVault
                and/or ProtocolSecretResolverMetrics registered in its service
                registry. These are resolved if available but not required.
            config: Resolver configuration specifying secret mappings, default TTL,
                and convention fallback settings. This is required because secret
                mappings are application-specific and cannot be auto-discovered
                from the container.

        Returns:
            Configured SecretResolver instance with container-resolved dependencies.

        Raises:
            ProtocolConfigurationError: If the container is invalid or missing
                the required ``service_registry`` attribute. The error includes
                :class:`ModelInfraErrorContext` with correlation_id for tracing.

        Example:
            Basic usage with container-resolved dependencies::

                container = ModelONEXContainer()
                await wire_infrastructure_services(container)

                config = ModelSecretResolverConfig(
                    mappings=[
                        ModelSecretMapping(
                            logical_name="database.password",
                            source=ModelSecretSourceSpec(
                                source_type=SecretSourceType.VAULT,
                                path="secret/myapp/db#password",
                            ),
                        ),
                    ],
                )
                resolver = await SecretResolver.from_container(container, config)
                password = await resolver.get_secret_async("database.password")

        Example (Container Without Optional Services):
            If HandlerVault is not registered, Vault-sourced secrets will fail
            at resolution time (not at factory creation)::

                container = ModelONEXContainer()
                # No wire_infrastructure_services() call - no Vault handler
                resolver = await SecretResolver.from_container(container, config)
                # Works for env/file secrets, fails for Vault secrets

        Note:
            **Why config is a required parameter:**

            Unlike other services that can be fully configured via container
            registration, SecretResolver requires explicit secret mappings that
            are application-specific. The config defines:

            - Which logical names map to which secret sources
            - Default TTL for caching
            - Whether convention fallback is enabled

            These settings vary per application and cannot be auto-discovered
            from the container's service registry.

        See Also:
            - :meth:`__init__`: Direct constructor for standalone usage
            - :class:`ModelSecretResolverConfig`: Configuration model
            - :class:`HandlerVault`: Vault handler for Vault-sourced secrets
        """
        from omnibase_infra.handlers.handler_vault import HandlerVault

        correlation_id = generate_correlation_id()

        # Validate container has service_registry
        if not hasattr(container, "service_registry"):
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="from_container",
                target_name="SecretResolver",
            )
            raise ProtocolConfigurationError(
                "Container missing required 'service_registry' attribute. "
                "Ensure container is a valid ModelONEXContainer instance.",
                context=context,
            )

        # Try to resolve optional dependencies from container
        vault_handler: HandlerVault | None = None
        metrics_collector: ProtocolSecretResolverMetrics | None = None

        # Attempt to resolve HandlerVault (optional)
        try:
            vault_handler = await container.service_registry.resolve_service(
                HandlerVault
            )
            logger.debug(
                "Resolved HandlerVault from container",
                extra={"correlation_id": str(correlation_id)},
            )
        except Exception as e:
            # HandlerVault not registered - this is acceptable
            logger.debug(
                "HandlerVault not available in container, Vault secrets disabled: %s",
                type(e).__name__,
                extra={"correlation_id": str(correlation_id)},
            )

        # Attempt to resolve metrics collector (optional)
        # Note: We use a broad try/except since the protocol may not be registered
        try:
            metrics_collector = await container.service_registry.resolve_service(
                ProtocolSecretResolverMetrics  # type: ignore[type-abstract]
            )
            logger.debug(
                "Resolved ProtocolSecretResolverMetrics from container",
                extra={"correlation_id": str(correlation_id)},
            )
        except Exception as e:
            # Metrics collector not registered - this is acceptable
            logger.debug(
                "Metrics collector not available in container: %s",
                type(e).__name__,
                extra={"correlation_id": str(correlation_id)},
            )

        return cls(
            config=config,
            vault_handler=vault_handler,
            metrics_collector=metrics_collector,
        )

    # === Primary API (Sync) ===

    def get_secret(
        self,
        logical_name: str,
        required: bool = True,
        correlation_id: UUID | None = None,
    ) -> SecretStr | None:
        """Resolve a secret by logical name.

        Resolution order:
            1. Check cache (if not expired)
            2. Try explicit mapping
            3. Try convention fallback (if enabled)
            4. Raise or return None based on required flag

        Warning:
            This synchronous method cannot resolve Vault secrets from within
            an async context (e.g., from inside an async function or coroutine).
            If you need to resolve Vault secrets in async code, use
            ``get_secret_async()`` instead. Calling this method from async
            context when resolving Vault secrets will raise SecretResolutionError.

        Args:
            logical_name: Dotted path (e.g., "database.postgres.password")
            required: If True, raises SecretResolutionError when not found
            correlation_id: Optional correlation ID for distributed tracing.
                If provided, propagates to error context for debugging.

        Returns:
            SecretStr if found, None if not found and required=False

        Raises:
            SecretResolutionError: If required=True and secret not found
        """
        # Generate correlation ID if not provided (for metrics/logging)
        effective_correlation_id = correlation_id or uuid4()

        with self._lock:
            # Check cache first
            cached = self._get_from_cache(logical_name)
            if cached is not None:
                self._record_resolution_success(
                    logical_name, "cache", effective_correlation_id
                )
                return cached

            # Resolve from source
            result = self._resolve_secret(logical_name, effective_correlation_id)

            if result is None:
                self._misses += 1
                if required:
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=effective_correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="get_secret",
                        target_name="secret_resolver",
                    )
                    # SECURITY: Log at DEBUG level only to avoid exposing secret identifiers
                    # in error messages shown to users/logs
                    logger.debug(
                        "Secret not found (correlation_id=%s): %s",
                        context.correlation_id,
                        logical_name,
                        extra={
                            "correlation_id": str(context.correlation_id),
                            "logical_name": logical_name,
                        },
                    )
                    # SECURITY: Do NOT include logical_name in error - it exposes secret identifiers
                    # Use correlation_id to trace back to DEBUG logs if needed
                    raise SecretResolutionError(
                        f"Required secret not found. "
                        f"See logs with correlation_id={context.correlation_id} for details.",
                        context=context,
                        # NOTE: Intentionally NOT passing logical_name to avoid exposing
                        # secret identifiers in error messages, logs, or serialized responses.
                    )
                return None

            return result

    def get_secrets(
        self,
        logical_names: list[str],
        required: bool = True,
        correlation_id: UUID | None = None,
    ) -> dict[str, SecretStr | None]:
        """Resolve multiple secrets.

        Args:
            logical_names: List of dotted paths
            required: If True, raises on first missing secret
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            Dict mapping logical_name -> SecretStr | None

        Note:
            This sync method resolves secrets sequentially. For better latency
            when resolving multiple secrets that involve I/O (Vault, file-based),
            prefer using ``get_secrets_async()`` which resolves in parallel via
            ``asyncio.gather()``.
        """
        return {
            name: self.get_secret(
                name, required=required, correlation_id=correlation_id
            )
            for name in logical_names
        }

    # === Primary API (Async) ===

    async def get_secret_async(
        self,
        logical_name: str,
        required: bool = True,
        correlation_id: UUID | None = None,
    ) -> SecretStr | None:
        """Async wrapper for get_secret.

        For Vault secrets, this uses async I/O. For env/file secrets,
        this wraps the sync call in a thread executor.

        Thread Safety:
            Uses threading.RLock for cache access to prevent race conditions
            with sync callers. Per-key async locks serialize resolution for the
            same secret while allowing parallel fetches for different secrets.

        Args:
            logical_name: Dotted path (e.g., "database.postgres.password")
            required: If True, raises SecretResolutionError when not found
            correlation_id: Optional correlation ID for distributed tracing.
                If provided, propagates to error context for debugging.

        Returns:
            SecretStr if found, None if not found and required=False

        Raises:
            SecretResolutionError: If required=True and secret not found
        """
        # Generate correlation ID if not provided (for metrics/logging)
        effective_correlation_id = correlation_id or uuid4()

        # Use threading lock for cache check (fast operation, prevents race with sync)
        with self._lock:
            cached = self._get_from_cache(logical_name)
            if cached is not None:
                self._record_resolution_success(
                    logical_name, "cache", effective_correlation_id
                )
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
                    self._record_resolution_success(
                        logical_name, "cache", effective_correlation_id
                    )
                    return cached

            # Resolve from source (potentially async for Vault)
            # Note: _resolve_secret_async handles its own locking for cache writes
            result = await self._resolve_secret_async(
                logical_name, effective_correlation_id
            )

            if result is None:
                with self._lock:
                    self._misses += 1
                if required:
                    context = ModelInfraErrorContext.with_correlation(
                        correlation_id=effective_correlation_id,
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation="get_secret_async",
                        target_name="secret_resolver",
                    )
                    # SECURITY: Log at DEBUG level only to avoid exposing secret identifiers
                    # in error messages shown to users/logs
                    logger.debug(
                        "Secret not found (correlation_id=%s): %s",
                        context.correlation_id,
                        logical_name,
                        extra={
                            "correlation_id": str(context.correlation_id),
                            "logical_name": logical_name,
                        },
                    )
                    # SECURITY: Do NOT include logical_name in error - it exposes secret identifiers
                    # Use correlation_id to trace back to DEBUG logs if needed
                    raise SecretResolutionError(
                        f"Required secret not found. "
                        f"See logs with correlation_id={context.correlation_id} for details.",
                        context=context,
                        # NOTE: Intentionally NOT passing logical_name to avoid exposing
                        # secret identifiers in error messages, logs, or serialized responses.
                    )
                return None

            return result

    def _maybe_log_eviction_warning(self, evict_count: int) -> None:
        """Log eviction warning with rate limiting (max once per minute).

        Prevents log flooding in high-throughput scenarios where evictions
        may occur frequently. The warning is only emitted if sufficient time
        has passed since the last warning.

        Args:
            evict_count: Number of entries that were evicted.

        Note:
            Uses time.monotonic() for reliable elapsed time measurement
            even if system clock changes.
        """
        current_time = time.monotonic()
        if (
            current_time - self._last_eviction_warning_time
            >= EVICTION_WARNING_INTERVAL_SECONDS
        ):
            self._last_eviction_warning_time = current_time
            logger.warning(
                "Async key locks at capacity (%d). Evicted %d oldest entries. "
                "This may indicate a DoS attack or dynamic logical name generation. "
                "Consider validating logical names against configured mappings. "
                "(Rate-limited: max 1 warning per minute)",
                MAX_ASYNC_KEY_LOCKS,
                evict_count,
                extra={
                    "max_locks": MAX_ASYNC_KEY_LOCKS,
                    "evicted_count": evict_count,
                    "current_count": len(self._async_key_locks),
                },
            )

    def _get_async_key_lock(self, logical_name: str) -> asyncio.Lock:
        """Get or create an async lock for a specific logical_name.

        This enables parallel resolution of different secrets while preventing
        duplicate concurrent fetches for the same secret.

        Thread Safety:
            Uses threading.RLock to safely access the key locks dictionary,
            ensuring thread-safe creation of new locks.

        LRU Eviction (Memory Leak Prevention):
            The ``_async_key_locks`` dictionary implements LRU eviction to prevent
            unbounded memory growth. When the dictionary reaches ``MAX_ASYNC_KEY_LOCKS``
            entries:

            1. The oldest 10% of entries are evicted (based on insertion order)
            2. A warning is logged indicating potential DoS or misconfiguration
            3. The new lock is then added

            This ensures memory usage is bounded while maintaining correctness:
            - Evicted locks are for secrets that were resolved earlier
            - If a secret is resolved again, a new lock will be created
            - The worst case is a brief period of duplicate concurrent fetches
              for recently-evicted secrets, which is acceptable (same value resolved)

        DoS Mitigation:
            - Warning threshold at ``ASYNC_KEY_LOCKS_WARNING_THRESHOLD`` for early detection
            - Hard cap at ``MAX_ASYNC_KEY_LOCKS`` with LRU eviction
            - Repeated eviction warnings indicate potential attack or misconfiguration
            - Validate logical names against configured mappings to prevent abuse

        Args:
            logical_name: The secret key to get a lock for

        Returns:
            asyncio.Lock for the given logical_name
        """
        with self._lock:
            if logical_name not in self._async_key_locks:
                lock_count = len(self._async_key_locks)

                # DoS mitigation: warn at threshold for early detection
                if lock_count == ASYNC_KEY_LOCKS_WARNING_THRESHOLD:
                    logger.warning(
                        "Async key locks dictionary reached %d entries - potential DoS risk. "
                        "Validate logical names against configured mappings.",
                        lock_count,
                        extra={"lock_count": lock_count},
                    )

                # LRU eviction: when at capacity, evict oldest 10% of entries
                if lock_count >= MAX_ASYNC_KEY_LOCKS:
                    evict_count = max(1, MAX_ASYNC_KEY_LOCKS // 10)  # 10%, minimum 1
                    # Python 3.7+ dicts maintain insertion order, so first keys are oldest
                    keys_to_evict = list(self._async_key_locks.keys())[:evict_count]
                    for key in keys_to_evict:
                        del self._async_key_locks[key]

                    # Rate-limited warning (max 1 per minute) to prevent log flooding
                    self._maybe_log_eviction_warning(evict_count)

                self._async_key_locks[logical_name] = asyncio.Lock()

            return self._async_key_locks[logical_name]

    async def get_secrets_async(
        self,
        logical_names: list[str],
        required: bool = True,
        correlation_id: UUID | None = None,
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
            required: If True, aggregates all failures into a single error
            correlation_id: Optional correlation ID for distributed tracing.

        Returns:
            Dict mapping logical_name -> SecretStr | None

        Raises:
            SecretResolutionError: If required=True and any secret is not found.
                All secrets are attempted before raising, and all failures are
                reported in a single aggregated error message.
        """
        if not logical_names:
            return {}

        # Create tasks for parallel resolution
        tasks = [
            self.get_secret_async(
                name, required=required, correlation_id=correlation_id
            )
            for name in logical_names
        ]

        # Gather results with return_exceptions=True for better error aggregation
        # This ensures all secrets are attempted before any error is raised
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for and aggregate exceptions
        failed_secrets: list[str] = []
        successful_results: dict[str, SecretStr | None] = {}

        for name, result in zip(logical_names, results, strict=True):
            if isinstance(result, BaseException):
                failed_secrets.append(name)
            else:
                successful_results[name] = result

        # If there were failures and required=True, raise aggregated error
        if failed_secrets and required:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="get_secrets_async",
                target_name="secret_resolver",
            )
            # SECURITY: Log at INFO level with count only (for operational awareness)
            # Secret identifiers are logged at DEBUG level only to prevent exposure
            logger.info(
                "Secret resolution failed for %d secret(s) (correlation_id=%s). "
                "Enable DEBUG logging to see secret names.",
                len(failed_secrets),
                context.correlation_id,
                extra={
                    "correlation_id": str(context.correlation_id),
                    "failed_count": len(failed_secrets),
                },
            )
            # SECURITY: Log failed secret names at DEBUG level only (with correlation_id)
            # to avoid exposing secret structure in error messages shown to users/logs
            logger.debug(
                "Failed secret names (correlation_id=%s): %s",
                context.correlation_id,
                ", ".join(failed_secrets),
                extra={
                    "correlation_id": str(context.correlation_id),
                    "failed_count": len(failed_secrets),
                },
            )
            # SECURITY: Do NOT include logical_name in error - it exposes secret identifiers
            # Use correlation_id to trace back to DEBUG logs if needed
            raise SecretResolutionError(
                f"Failed to resolve {len(failed_secrets)} secret(s). "
                f"See logs with correlation_id={context.correlation_id} for details.",
                context=context,
                # NOTE: Intentionally NOT passing logical_name to avoid exposing
                # secret identifiers in error messages, logs, or serialized responses.
                # The correlation_id can be used to find secret names in DEBUG logs.
            )

        return successful_results

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

    def get_resolution_metrics(self) -> ModelSecretResolverMetrics:
        """Return resolution metrics for observability.

        Returns:
            ModelSecretResolverMetrics with:
                - success_counts: Dict of source_type -> success count
                - failure_counts: Dict of source_type -> failure count
                - latency_samples: Number of latency samples collected
                - avg_latency_ms: Average resolution latency (if samples > 0)
                - cache_hits: Total number of cache hits
                - cache_misses: Total number of cache misses
        """
        with self._lock:
            avg_latency = 0.0
            if self._resolution_latencies:
                avg_latency = sum(lat for _, lat in self._resolution_latencies) / len(
                    self._resolution_latencies
                )

            return ModelSecretResolverMetrics(
                success_counts=dict(self._resolution_success_counts),
                failure_counts=dict(self._resolution_failure_counts),
                latency_samples=len(self._resolution_latencies),
                avg_latency_ms=avg_latency,
                cache_hits=self._hits,
                cache_misses=self._misses,
            )

    def set_metrics_collector(
        self, collector: ProtocolSecretResolverMetrics | None
    ) -> None:
        """Set the external metrics collector.

        Thread Safety:
            This method is thread-safe. The collector reference is updated
            atomically under ``_lock``. Concurrent calls to resolution methods
            will see either the old or new collector (never a partial state).

            The pattern used in ``_record_resolution_success`` and
            ``_record_resolution_failure`` captures the collector reference
            while holding the lock, then uses it outside the lock. This ensures
            that even if ``set_metrics_collector()`` is called concurrently,
            each resolution operation uses a consistent collector reference.

        Args:
            collector: Metrics collector implementing ProtocolSecretResolverMetrics,
                or None to disable external metrics collection.
        """
        with self._lock:
            self._metrics_collector = collector

    def _record_resolution_success(
        self,
        logical_name: str,
        source_type: str,
        correlation_id: UUID,
        start_time: float | None = None,
    ) -> None:
        """Record a successful secret resolution.

        Thread Safety:
            Captures ``_metrics_collector`` reference while holding ``_lock`` to
            prevent race conditions with concurrent ``set_metrics_collector()``
            calls. The captured reference is then used outside the lock to avoid
            holding the lock during potentially slow I/O operations.

        Args:
            logical_name: The secret's logical name
            source_type: Source type (env, file, vault, cache)
            correlation_id: Correlation ID for tracing
            start_time: Optional start time from time.monotonic() for latency calc
        """
        latency_ms = 0.0
        if start_time is not None:
            latency_ms = (time.monotonic() - start_time) * 1000

        # Internal tracking + capture collector reference atomically
        with self._lock:
            self._resolution_success_counts[source_type] += 1
            if start_time is not None:
                # deque with maxlen=1000 automatically rotates (O(1) vs O(n) for list.pop(0))
                self._resolution_latencies.append((source_type, latency_ms))
            # THREAD SAFETY: Capture collector reference while holding lock to prevent
            # race with set_metrics_collector(). Use captured ref outside lock.
            collector = self._metrics_collector

        # External metrics collector - use captured reference (may be None)
        if collector is not None:
            try:
                if hasattr(collector, "record_resolution_success"):
                    collector.record_resolution_success(source_type)
                if start_time is not None and hasattr(
                    collector, "record_resolution_latency"
                ):
                    collector.record_resolution_latency(source_type, latency_ms)
                if source_type == "cache" and hasattr(collector, "record_cache_hit"):
                    collector.record_cache_hit()
            except Exception as e:
                # Never let metrics failures affect secret resolution, but log
                # at warning level since a configured collector failing indicates
                # an integration issue worth investigating.
                logger.warning(
                    "Metrics collector error (ignored, resolution unaffected): %s",
                    e,
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(correlation_id),
                        "exception_type": type(e).__name__,
                    },
                )

        # Structured logging
        logger.debug(
            "Secret resolved successfully: %s",
            logical_name,
            extra={
                "logical_name": logical_name,
                "source_type": source_type,
                "cache_hit": source_type == "cache",
                "latency_ms": latency_ms,
                "correlation_id": str(correlation_id),
            },
        )

    def _record_resolution_failure(
        self,
        logical_name: str,
        source_type: str,
        correlation_id: UUID,
        reason: str,
    ) -> None:
        """Record a failed secret resolution.

        Thread Safety:
            Captures ``_metrics_collector`` reference while holding ``_lock`` to
            prevent race conditions with concurrent ``set_metrics_collector()``
            calls. The captured reference is then used outside the lock to avoid
            holding the lock during potentially slow I/O operations.

        Args:
            logical_name: The secret's logical name
            source_type: Source type (env, file, vault, unknown)
            correlation_id: Correlation ID for tracing
            reason: Failure reason (not_found, handler_not_configured, etc.)
        """
        # Internal tracking + capture collector reference atomically
        with self._lock:
            self._resolution_failure_counts[source_type] += 1
            # THREAD SAFETY: Capture collector reference while holding lock to prevent
            # race with set_metrics_collector(). Use captured ref outside lock.
            collector = self._metrics_collector

        # External metrics collector - use captured reference (may be None)
        if collector is not None:
            try:
                if hasattr(collector, "record_resolution_failure"):
                    collector.record_resolution_failure(source_type)
                # NOTE: Do NOT call record_cache_miss() here - resolution failures
                # are distinct from cache misses. Cache misses are already tracked
                # in get_secret() and get_secret_async() via self._misses += 1
            except Exception as e:
                # Never let metrics failures affect secret resolution, but log
                # at warning level since a configured collector failing indicates
                # an integration issue worth investigating.
                logger.warning(
                    "Metrics collector error (ignored, resolution unaffected): %s",
                    e,
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(correlation_id),
                        "exception_type": type(e).__name__,
                    },
                )

        # Structured logging - level depends on failure type
        # Configuration issues (no_mapping, handler_not_configured) are warnings
        # since they indicate misconfiguration that should be addressed.
        # Expected failures (not_found for optional secrets) are debug level.
        if reason in ("no_mapping", "handler_not_configured"):
            logger.warning(
                "Secret resolution failed (configuration issue): %s",
                logical_name,
                extra={
                    "logical_name": logical_name,
                    "source_type": source_type,
                    "reason": reason,
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            # not_found and other expected failures - debug level
            logger.debug(
                "Secret resolution failed: %s",
                logical_name,
                extra={
                    "logical_name": logical_name,
                    "source_type": source_type,
                    "reason": reason,
                    "correlation_id": str(correlation_id),
                },
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
            They are ALWAYS resolved from environment variables only (never vault/file).
            If an explicit mapping exists for an env source, that mapping is honored.
            Otherwise, convention-based naming (logical_name -> ENV_VAR) is used.

        Args:
            logical_name: The bootstrap secret's logical name

        Returns:
            SecretStr if found, None if env var is not set
        """
        # First, check for explicit env var mapping (same priority as normal secrets)
        # This ensures that explicit mappings like:
        #   {"vault.token": ModelSecretSourceSpec(source_type="env", source_path="MY_VAULT_TOKEN")}
        # are respected for bootstrap secrets.
        if logical_name in self._mappings:
            mapping = self._mappings[logical_name]
            if mapping.source_type == "env":
                # Use the explicitly mapped env var name
                env_var = mapping.source_path
            else:
                # Non-env mappings (vault/file) are invalid for bootstrap secrets
                # by design - they must come from env to avoid circular dependency.
                # Fall back to convention for the env var name.
                env_var = self._logical_name_to_env_var(logical_name)
        else:
            # No explicit mapping - use convention fallback
            env_var = self._logical_name_to_env_var(logical_name)

        value = os.environ.get(env_var)

        if value is None:
            return None

        return SecretStr(value)

    def _resolve_secret(
        self, logical_name: str, correlation_id: UUID | None = None
    ) -> SecretStr | None:
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
            correlation_id: Optional correlation ID for tracing

        Returns:
            SecretStr if found, None otherwise
        """
        effective_correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        # SECURITY: Bootstrap secrets bypass normal resolution
        # They must come from env vars to avoid circular dependency with Vault
        if self._is_bootstrap_secret(logical_name):
            secret = self._resolve_bootstrap_secret_value(logical_name)
            if secret is not None:
                # Cache write is safe here - caller holds _lock
                self._cache_secret(logical_name, secret, "env")
                self._record_resolution_success(
                    logical_name, "env", effective_correlation_id, start_time
                )
            return secret

        source = self._get_source_spec(logical_name)
        if source is None:
            self._record_resolution_failure(
                logical_name, "unknown", effective_correlation_id, "no_mapping"
            )
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
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(effective_correlation_id),
                    },
                )
                self._record_resolution_failure(
                    logical_name,
                    "vault",
                    effective_correlation_id,
                    "handler_not_configured",
                )
                return None
            value = self._read_vault_secret_sync(
                source.source_path, logical_name, effective_correlation_id
            )

        if value is None:
            self._record_resolution_failure(
                logical_name, source.source_type, effective_correlation_id, "not_found"
            )
            return None

        secret = SecretStr(value)
        self._cache_secret(logical_name, secret, source.source_type)
        self._record_resolution_success(
            logical_name, source.source_type, effective_correlation_id, start_time
        )
        return secret

    async def _resolve_secret_async(
        self, logical_name: str, correlation_id: UUID | None = None
    ) -> SecretStr | None:
        """Resolve secret from source asynchronously.

        Thread Safety:
            Uses threading.RLock for cache writes to prevent race conditions
            with sync callers. I/O operations are performed outside the lock.
            Bootstrap secrets also use _lock for their cache writes to ensure
            thread-safe access from both sync and async contexts.

        Security:
            Bootstrap secrets (vault.token, vault.addr, etc.) are resolved directly
            from environment variables, bypassing the normal source chain. This
            prevents circular dependencies when initializing Vault.

        Args:
            logical_name: The logical name to resolve
            correlation_id: Optional correlation ID for tracing

        Returns:
            SecretStr if found, None otherwise
        """
        effective_correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

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
                self._record_resolution_success(
                    logical_name, "env", effective_correlation_id, start_time
                )
            return secret

        source = self._get_source_spec(logical_name)
        if source is None:
            self._record_resolution_failure(
                logical_name, "unknown", effective_correlation_id, "no_mapping"
            )
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
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(effective_correlation_id),
                    },
                )
                self._record_resolution_failure(
                    logical_name,
                    "vault",
                    effective_correlation_id,
                    "handler_not_configured",
                )
                return None
            value = await self._read_vault_secret_async(
                source.source_path, logical_name, effective_correlation_id
            )

        if value is None:
            self._record_resolution_failure(
                logical_name, source.source_type, effective_correlation_id, "not_found"
            )
            return None

        secret = SecretStr(value)
        # THREAD SAFETY: Use lock for cache write to prevent race with sync callers
        # Check-before-write pattern avoids unnecessary overwrites if sync caller
        # already cached this secret between our cache check and now
        with self._lock:
            if logical_name not in self._cache:
                self._cache_secret(logical_name, secret, source.source_type)
        self._record_resolution_success(
            logical_name, source.source_type, effective_correlation_id, start_time
        )
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
            # Use is_relative_to() to check without raising (Python 3.9+)
            if not resolved_path.is_relative_to(secrets_dir_resolved):
                # Path escapes secrets_dir - this is a path traversal attempt
                # SECURITY: Log at ERROR level - potential attack indicator
                logger.error(
                    "Path traversal detected for secret: %s",
                    logical_name,
                    extra={"logical_name": logical_name},
                )
                return None

        # Avoid TOCTOU race: read atomically with size limit instead of stat() then read()
        # This prevents an attacker from swapping the file between size check and read
        try:
            # Read up to MAX_SECRET_FILE_SIZE + 1 bytes atomically
            # If we got more than MAX_SECRET_FILE_SIZE, the file is too large
            with resolved_path.open("r") as f:
                content = f.read(MAX_SECRET_FILE_SIZE + 1)
                if len(content) > MAX_SECRET_FILE_SIZE:
                    logger.warning(
                        "Secret file exceeds size limit: %s",
                        logical_name,
                        extra={"logical_name": logical_name},
                    )
                    return None
                return content.strip()
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

    def _read_vault_secret_sync(
        self, path: str, logical_name: str = "", correlation_id: UUID | None = None
    ) -> str | None:
        """Read secret from Vault synchronously.

        This method wraps the async Vault handler for synchronous contexts.
        It creates a new event loop if one is not running, otherwise raises
        an error (cannot nest event loops).

        Path format: "mount/path#field" or "mount/path" (returns first field value)

        Security:
            - This method never logs Vault paths (could reveal secret structure)
            - Secret values are never logged at any level
            - Error messages are sanitized to include only logical names

        Args:
            path: Vault path with optional field specifier
            logical_name: The logical name being resolved (for error context only)
            correlation_id: Optional correlation ID for tracing

        Returns:
            Secret value or None if not found

        Raises:
            SecretResolutionError: On Vault communication failures or if called
                from within an async context (cannot nest event loops)
            InfraAuthenticationError: If authentication fails
            InfraTimeoutError: If the request times out
            InfraUnavailableError: If Vault is unavailable (circuit breaker open)
        """
        if self._vault_handler is None:
            return None

        effective_correlation_id = correlation_id or uuid4()

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - cannot use asyncio.run()
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="read_secret_sync",
                target_name="secret_resolver",
            )
            raise SecretResolutionError(
                f"Cannot resolve Vault secret synchronously from async context: "
                f"{logical_name}. Use get_secret_async() instead.",
                context=context,
                logical_name=logical_name,
            )
        except RuntimeError:
            # No running event loop - safe to use asyncio.run()
            pass

        # Run the async method in a new event loop
        return asyncio.run(
            self._read_vault_secret_async(path, logical_name, effective_correlation_id)
        )

    async def _read_vault_secret_async(
        self, path: str, logical_name: str = "", correlation_id: UUID | None = None
    ) -> str | None:
        """Read secret from Vault asynchronously.

        Path format: "mount/path#field" or "mount/path" (returns first field value)

        Examples:
            "secret/myapp/db#password" -> mount="secret", path="myapp/db", field="password"
            "secret/myapp/db" -> mount="secret", path="myapp/db", field=None (first value)

        Type Handling:
            All Vault values are converted to strings via ``str()``. This is intentional
            because SecretResolver returns ``SecretStr`` values, which only wrap strings.
            Non-string Vault values (integers, booleans, dicts) are converted as follows:

            - Integers: ``123`` -> ``"123"``
            - Booleans: ``True`` -> ``"True"``
            - Lists/Dicts: Python repr (NOT JSON) - avoid storing complex types

            Best Practice: Store secrets as strings in Vault. If you need structured
            data, store it as a JSON string and parse after resolution.

        Security:
            - This method never logs Vault paths (could reveal secret structure)
            - Secret values are never logged at any level
            - Error messages are sanitized to include only logical names

        Args:
            path: Vault path with optional field specifier (mount/path#field)
            logical_name: The logical name being resolved (for error context only)
            correlation_id: Optional correlation ID for tracing

        Returns:
            Secret value as string, or None if not found

        Raises:
            SecretResolutionError: On Vault communication failures
            InfraAuthenticationError: If authentication fails
            InfraTimeoutError: If the request times out
            InfraUnavailableError: If Vault is unavailable (circuit breaker open)
        """
        if self._vault_handler is None:
            return None

        effective_correlation_id = correlation_id or uuid4()

        # Parse path into mount_point, vault_path, and optional field
        mount_point, vault_path, field = self._parse_vault_path_components(path)

        # Create envelope for vault.read_secret operation
        envelope: JsonType = {
            "operation": "vault.read_secret",
            "payload": {
                "path": vault_path,
                "mount_point": mount_point,
            },
            "correlation_id": str(effective_correlation_id),
        }

        try:
            result = await self._vault_handler.execute(
                cast("dict[str, object]", envelope)
            )

            # Extract secret data from handler response
            # Response format: {"status": "success", "payload": {"data": {...}, "metadata": {...}}}
            result_dict = result.result
            if not isinstance(result_dict, dict):
                logger.warning(
                    "Unexpected Vault response format for secret: %s",
                    logical_name,
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(effective_correlation_id),
                    },
                )
                return None

            status = result_dict.get("status")
            if status != "success":
                logger.debug(
                    "Vault returned non-success status for secret: %s",
                    logical_name,
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(effective_correlation_id),
                    },
                )
                return None

            payload = result_dict.get("payload", {})
            if not isinstance(payload, dict):
                return None

            secret_data = payload.get("data", {})
            if not isinstance(secret_data, dict) or not secret_data:
                logger.debug(
                    "No secret data found in Vault for: %s",
                    logical_name,
                    extra={
                        "logical_name": logical_name,
                        "correlation_id": str(effective_correlation_id),
                    },
                )
                return None

            # Extract the specific field or first value
            if field:
                value = secret_data.get(field)
                if value is None:
                    logger.debug(
                        "Field not found in Vault secret: %s",
                        logical_name,
                        extra={
                            "logical_name": logical_name,
                            "correlation_id": str(effective_correlation_id),
                        },
                    )
                    return None
            else:
                # No field specified - return first value
                value = next(iter(secret_data.values()), None)

            if value is None:
                return None

            # SECURITY: String conversion is INTENTIONAL for SecretStr compatibility.
            #
            # SecretStr (from Pydantic) only wraps string values to prevent accidental
            # logging of secrets. Since SecretResolver returns SecretStr, all Vault
            # values must be converted to strings.
            #
            # Type conversion behavior:
            #   - Strings: returned as-is
            #   - Integers: "123" (str representation)
            #   - Booleans: "True" or "False" (Python str representation)
            #   - Lists/Dicts: Python repr (NOT JSON) - avoid storing complex types
            #
            # Best Practice: Store secrets as strings in Vault. If you need structured
            # data, store it as a JSON string and parse after resolution.
            return str(value)

        except InfraAuthenticationError:
            # Re-raise auth errors directly - they have proper context
            raise
        except InfraTimeoutError:
            # Re-raise timeout errors directly - they have proper context
            raise
        except InfraUnavailableError:
            # Re-raise unavailable errors (circuit breaker open)
            raise
        except Exception as e:
            # Wrap other errors in SecretResolutionError with sanitized message
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=effective_correlation_id,
                transport_type=EnumInfraTransportType.VAULT,
                operation="read_secret",
                target_name="secret_resolver",
            )
            raise SecretResolutionError(
                f"Failed to resolve secret from Vault: {logical_name}",
                context=context,
                logical_name=logical_name,
            ) from e

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

    def _parse_vault_path_components(self, path: str) -> tuple[str, str, str | None]:
        """Parse Vault path into mount_point, path, and optional field.

        The path format is: "mount_point/path/to/secret#field"

        For KV v2 secrets engine, the path convention is:
            - mount_point: The secrets engine mount (e.g., "secret")
            - path: The secret path within the mount (e.g., "myapp/db")
            - field: Optional specific field to extract (e.g., "password")

        Examples:
            "secret/myapp/db#password" -> ("secret", "myapp/db", "password")
            "secret/myapp/db" -> ("secret", "myapp/db", None)
            "kv/prod/config#api_key" -> ("kv", "prod/config", "api_key")
            "secret#password" -> ("secret", "", "password")  # Edge case - unusual format

        Args:
            path: Full Vault path with optional field specifier

        Returns:
            Tuple of (mount_point, vault_path, field_name or None)
        """
        # First extract field if present
        if "#" in path:
            path_without_field, field = path.rsplit("#", 1)
        else:
            path_without_field = path
            field = None

        # Split into mount_point and rest of path
        # First component is always the mount_point
        parts = path_without_field.split("/", 1)
        if len(parts) == 1:
            # Edge case: No slash in path - entire path is treated as mount_point
            # with empty vault_path. This format (e.g., "secret#field") is unusual
            # and may not be supported by Vault's KV v2 engine which expects
            # paths like "mount/path". Log a warning to alert operators.
            logger.warning(
                "Unusual Vault path format detected. "
                "Expected 'mount/path#field' format with at least one '/' separator. "
                "Empty path segment may not work with Vault KV v2.",
            )
            return parts[0], "", field

        mount_point = parts[0]
        vault_path = parts[1]

        return mount_point, vault_path, field

    def _cache_secret(
        self,
        logical_name: str,
        value: SecretStr,
        source_type: SecretSourceType,
    ) -> None:
        """Cache a resolved secret with appropriate TTL and jitter.

        TTL Jitter:
            A symmetric random jitter of ±10% is added to the base TTL to
            prevent cache stampede scenarios where many cached entries expire
            at the same time, causing a thundering herd of resolution requests.
            The symmetric distribution provides better spread than additive-only
            jitter, reducing the probability of clustered expirations.

        Args:
            logical_name: The logical name being cached
            value: The secret value to cache
            source_type: Source type for TTL selection
        """
        base_ttl_seconds = self._get_ttl(logical_name, source_type)
        # Add ±10% jitter to prevent cache stampede (thundering herd)
        jitter_factor = random.uniform(
            -CACHE_TTL_JITTER_PERCENT, CACHE_TTL_JITTER_PERCENT
        )
        ttl_seconds = max(1, int(base_ttl_seconds * (1 + jitter_factor)))
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


__all__: list[str] = [
    "ProtocolSecretResolverMetrics",
    "SecretResolver",
    "SecretSourceType",
]
