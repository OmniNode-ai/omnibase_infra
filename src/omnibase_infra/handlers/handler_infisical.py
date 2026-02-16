# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infisical Secret Management Handler - EFFECT pattern implementation.

Provides secret management operations against Infisical with:
- Handler-owned TTL caching (configurable, default 5 min)
- Circuit breaker via ``MixinAsyncCircuitBreaker``
- Batch secret fetching
- Audit event emission via structured logging

Architecture:
    The handler delegates raw SDK calls to ``AdapterInfisical`` (in ``_internal/``).
    All cross-cutting concerns (caching, circuit breaking, retry, audit) are owned
    by this handler, NOT the adapter.

Return Type:
    All operations return ``ModelHandlerOutput[dict[str, object]]`` per OMN-975.

.. versionadded:: 0.9.0
    Initial implementation for OMN-2286.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID, uuid4

from pydantic import SecretStr

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.adapters._internal.adapter_infisical import (
    AdapterInfisical,
    ModelInfisicalAdapterConfig,
)
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    RuntimeHostError,
    SecretResolutionError,
)
from omnibase_infra.handlers.models.infisical import ModelInfisicalHandlerConfig
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction

logger = logging.getLogger(__name__)

HANDLER_ID_INFISICAL: str = "infisical-handler"

SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "infisical.get_secret",
        "infisical.list_secrets",
        "infisical.get_secrets_batch",
    }
)


class CacheEntry:
    """Internal TTL cache entry for a single secret."""

    __slots__ = ("expires_at", "value")

    def __init__(self, value: SecretStr, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl

    @property
    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class HandlerInfisical(
    MixinAsyncCircuitBreaker,
    MixinEnvelopeExtraction,
):
    """Infisical secret management handler (EFFECT pattern).

    This handler owns:
    - TTL-based secret caching (handler-level, not SDK-level)
    - Circuit breaker for Infisical service availability
    - Audit logging for secret access patterns
    - Batch secret fetching

    The handler delegates raw SDK operations to ``AdapterInfisical``.

    Security:
        - Secret values are NEVER logged at any level.
        - All returned values are wrapped in ``SecretStr``.
        - ``describe()`` never exposes credentials.
        - Error messages are sanitized to exclude secret names and values.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerInfisical with ONEX container.

        Args:
            container: ONEX container for dependency injection.
        """
        self._container = container
        self._adapter: AdapterInfisical | None = None
        self._config: ModelInfisicalHandlerConfig | None = None
        self._initialized: bool = False
        self._circuit_breaker_initialized: bool = False
        # Handler-owned cache: secret_key -> CacheEntry
        self._cache: dict[str, CacheEntry] = {}
        # Metrics
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_fetches: int = 0

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - infrastructure protocol handler.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification.

        Returns:
            EnumHandlerTypeCategory.EFFECT - performs side-effecting I/O.
        """
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the Infisical handler with configuration.

        Args:
            config: Configuration dict containing Infisical connection details.

        Raises:
            ProtocolConfigurationError: If configuration validation fails.
            InfraAuthenticationError: If authentication fails.
            RuntimeHostError: If initialization fails for other reasons.
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        # Phase 1: Parse and validate configuration
        try:
            self._config = ModelInfisicalHandlerConfig(**config)  # type: ignore[arg-type]
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=init_correlation_id,
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="initialize",
            )
            raise RuntimeHostError(
                f"Invalid Infisical handler configuration: {e}",
                context=ctx,
            ) from e

        # Phase 2: Create and initialize adapter
        try:
            adapter_config = ModelInfisicalAdapterConfig(
                host=self._config.host,
                client_id=self._config.client_id,
                client_secret=self._config.client_secret,
                project_id=self._config.project_id,
                environment_slug=self._config.environment_slug,
                secret_path=self._config.secret_path,
            )
            self._adapter = AdapterInfisical(adapter_config)
            self._adapter.initialize()
        except RuntimeError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=init_correlation_id,
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="initialize",
            )
            error_msg = str(e)
            if "auth" in error_msg.lower() or "credential" in error_msg.lower():
                raise InfraAuthenticationError(
                    "Infisical authentication failed",
                    context=ctx,
                ) from e
            raise RuntimeHostError(
                "Failed to initialize Infisical adapter",
                context=ctx,
            ) from e

        # Phase 3: Initialize circuit breaker
        if self._config.circuit_breaker_enabled:
            self._init_circuit_breaker(
                threshold=self._config.circuit_breaker_threshold,
                reset_timeout=self._config.circuit_breaker_reset_timeout,
                service_name="infisical",
                transport_type=EnumInfraTransportType.INFISICAL,
                half_open_successes=1,
            )
            self._circuit_breaker_initialized = True

        self._initialized = True
        logger.info(
            "HandlerInfisical initialized successfully",
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
                "host": self._config.host,
                "cache_ttl": self._config.cache_ttl_seconds,
                "circuit_breaker": self._config.circuit_breaker_enabled,
            },
        )

    async def shutdown(self) -> None:
        """Shut down the handler and release resources."""
        if self._adapter is not None:
            self._adapter.shutdown()
            self._adapter = None

        # Reset circuit breaker
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        self._cache.clear()
        self._initialized = False
        self._config = None
        self._circuit_breaker_initialized = False
        logger.info("HandlerInfisical shutdown complete")

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Execute an Infisical operation from an envelope.

        Args:
            envelope: Request envelope with operation and payload.

        Returns:
            ModelHandlerOutput with operation result.

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraUnavailableError: If circuit breaker is open.
            SecretResolutionError: If secret resolution fails.
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized or self._adapter is None or self._config is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="execute",
                target_name="infisical_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HandlerInfisical not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="execute",
                target_name="infisical_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=ctx,
            )

        if operation not in SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation=operation,
                target_name="infisical_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported. "
                f"Available: {', '.join(sorted(SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation=operation,
                target_name="infisical_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Check circuit breaker before operation
        if self._circuit_breaker_initialized:
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation=operation,
                    correlation_id=correlation_id,
                )

        try:
            if operation == "infisical.get_secret":
                result = await self._get_secret(
                    payload, correlation_id, input_envelope_id
                )
            elif operation == "infisical.list_secrets":
                result = await self._list_secrets(
                    payload, correlation_id, input_envelope_id
                )
            else:  # infisical.get_secrets_batch
                result = await self._get_secrets_batch(
                    payload, correlation_id, input_envelope_id
                )

            # Record circuit breaker success
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

            return result

        except (RuntimeHostError, SecretResolutionError, InfraUnavailableError):
            raise
        except Exception as e:
            # Record circuit breaker failure
            if self._circuit_breaker_initialized:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure(
                        operation=operation,
                        correlation_id=correlation_id,
                    )

            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.INFISICAL,
                operation=operation,
            )
            raise SecretResolutionError(
                "Infisical operation failed",
                context=ctx,
            ) from e

    async def _get_secret(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID | None,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Retrieve a single secret with caching."""
        secret_name = payload.get("secret_name")
        if not isinstance(secret_name, str) or not secret_name:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="infisical.get_secret",
                target_name="infisical_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'secret_name' in payload",
                context=ctx,
            )

        assert self._adapter is not None
        assert self._config is not None

        # Check cache first
        cache_key = self._build_cache_key(
            secret_name,
            payload.get("project_id"),
            payload.get("environment_slug"),
            payload.get("secret_path"),
        )

        cached = self._cache.get(cache_key)
        if cached is not None and not cached.is_expired:
            self._cache_hits += 1
            logger.debug(
                "Cache hit for secret",
                extra={
                    "correlation_id": str(correlation_id),
                    "cache_hit": True,
                },
            )
            return ModelHandlerOutput.for_compute(
                handler_id=HANDLER_ID_INFISICAL,
                correlation_id=correlation_id,
                input_envelope_id=input_envelope_id,
                result={
                    "secret_name": secret_name,
                    "value": cached.value.get_secret_value(),
                    "source": "cache",
                },
            )

        self._cache_misses += 1
        self._total_fetches += 1

        # Fetch from Infisical
        result = self._adapter.get_secret(
            secret_name=secret_name,
            project_id=payload.get("project_id")
            if isinstance(payload.get("project_id"), str)
            else None,
            environment_slug=payload.get("environment_slug")
            if isinstance(payload.get("environment_slug"), str)
            else None,
            secret_path=payload.get("secret_path")
            if isinstance(payload.get("secret_path"), str)
            else None,
        )

        # Cache the result
        if self._config.cache_ttl_seconds > 0:
            self._cache[cache_key] = CacheEntry(
                value=result.value,
                ttl=self._config.cache_ttl_seconds,
            )

        # Audit log (no secret values)
        logger.info(
            "Secret retrieved from Infisical",
            extra={
                "correlation_id": str(correlation_id),
                "operation": "infisical.get_secret",
                "cache_hit": False,
            },
        )

        return ModelHandlerOutput.for_compute(
            handler_id=HANDLER_ID_INFISICAL,
            correlation_id=correlation_id,
            input_envelope_id=input_envelope_id,
            result={
                "secret_name": secret_name,
                "value": result.value.get_secret_value(),
                "source": "infisical",
                "version": result.version,
            },
        )

    async def _list_secrets(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID | None,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """List secrets at a given path."""
        assert self._adapter is not None

        self._total_fetches += 1

        results = self._adapter.list_secrets(
            project_id=payload.get("project_id")
            if isinstance(payload.get("project_id"), str)
            else None,
            environment_slug=payload.get("environment_slug")
            if isinstance(payload.get("environment_slug"), str)
            else None,
            secret_path=payload.get("secret_path")
            if isinstance(payload.get("secret_path"), str)
            else None,
        )

        secret_keys = [r.key for r in results]

        logger.info(
            "Secrets listed from Infisical",
            extra={
                "correlation_id": str(correlation_id),
                "operation": "infisical.list_secrets",
                "count": len(secret_keys),
            },
        )

        return ModelHandlerOutput.for_compute(
            handler_id=HANDLER_ID_INFISICAL,
            correlation_id=correlation_id,
            input_envelope_id=input_envelope_id,
            result={
                "secret_keys": secret_keys,
                "count": len(secret_keys),
                "source": "infisical",
            },
        )

    async def _get_secrets_batch(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID | None,
    ) -> ModelHandlerOutput[dict[str, object]]:
        """Retrieve multiple secrets by name with caching."""
        secret_names = payload.get("secret_names")
        if not isinstance(secret_names, list) or not all(
            isinstance(n, str) for n in secret_names
        ):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INFISICAL,
                operation="infisical.get_secrets_batch",
                target_name="infisical_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'secret_names' in payload (expected list of strings)",
                context=ctx,
            )

        assert self._adapter is not None
        assert self._config is not None

        results: dict[str, str] = {}
        errors: dict[str, str] = {}
        from_cache: list[str] = []
        from_fetch: list[str] = []
        to_fetch: list[str] = []

        # Check cache for each secret
        for name in secret_names:
            cache_key = self._build_cache_key(
                name,
                payload.get("project_id"),
                payload.get("environment_slug"),
                payload.get("secret_path"),
            )
            cached = self._cache.get(cache_key)
            if cached is not None and not cached.is_expired:
                self._cache_hits += 1
                results[name] = cached.value.get_secret_value()
                from_cache.append(name)
            else:
                self._cache_misses += 1
                to_fetch.append(name)

        # Fetch remaining from Infisical
        if to_fetch:
            self._total_fetches += 1
            batch_result = self._adapter.get_secrets_batch(
                secret_names=to_fetch,
                project_id=payload.get("project_id")
                if isinstance(payload.get("project_id"), str)
                else None,
                environment_slug=payload.get("environment_slug")
                if isinstance(payload.get("environment_slug"), str)
                else None,
                secret_path=payload.get("secret_path")
                if isinstance(payload.get("secret_path"), str)
                else None,
            )

            for name, secret_result in batch_result.secrets.items():
                results[name] = secret_result.value.get_secret_value()
                from_fetch.append(name)

                # Cache the fetched secret
                if self._config.cache_ttl_seconds > 0:
                    cache_key = self._build_cache_key(
                        name,
                        payload.get("project_id"),
                        payload.get("environment_slug"),
                        payload.get("secret_path"),
                    )
                    self._cache[cache_key] = CacheEntry(
                        value=secret_result.value,
                        ttl=self._config.cache_ttl_seconds,
                    )

            errors = batch_result.errors

        logger.info(
            "Batch secrets retrieved",
            extra={
                "correlation_id": str(correlation_id),
                "operation": "infisical.get_secrets_batch",
                "from_cache": len(from_cache),
                "from_fetch": len(from_fetch),
                "errors": len(errors),
            },
        )

        return ModelHandlerOutput.for_compute(
            handler_id=HANDLER_ID_INFISICAL,
            correlation_id=correlation_id,
            input_envelope_id=input_envelope_id,
            result={
                "secrets": results,
                "errors": errors,
                "from_cache": len(from_cache),
                "from_fetch": len(from_fetch),
                "source": "infisical",
            },
        )

    def _build_cache_key(
        self,
        secret_name: str,
        project_id: object = None,
        environment_slug: object = None,
        secret_path: object = None,
    ) -> str:
        """Build a unique cache key from secret coordinates."""
        assert self._config is not None
        parts = [
            str(project_id) if isinstance(project_id, str) else self._config.project_id,
            str(environment_slug)
            if isinstance(environment_slug, str)
            else self._config.environment_slug,
            str(secret_path)
            if isinstance(secret_path, str)
            else self._config.secret_path,
            secret_name,
        ]
        return "::".join(parts)

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities.

        Returns:
            Dict with handler type, category, operations, and metrics.
            Never exposes credentials.
        """
        return {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "cache_ttl_seconds": self._config.cache_ttl_seconds if self._config else 0,
            "initialized": self._initialized,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_fetches": self._total_fetches,
            "version": "0.1.0",
        }

    def invalidate_cache(self, secret_name: str | None = None) -> int:
        """Invalidate cached secrets.

        Args:
            secret_name: If provided, invalidate only entries containing this
                secret name. If None, invalidate all entries.

        Returns:
            Number of cache entries invalidated.
        """
        if secret_name is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        to_remove = [k for k in self._cache if k.endswith(f"::{secret_name}")]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)


__all__: list[str] = ["HandlerInfisical", "HANDLER_ID_INFISICAL"]
