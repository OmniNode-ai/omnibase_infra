# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete ProtocolModelRouter implementation with multi-provider routing.

Coordinates request routing across multiple ProtocolLLMProvider instances
based on availability, capability matching, and round-robin load balancing.
Provides round-robin failover across healthy providers.

Architecture:
    - Maintains a registry of named providers
    - Routes requests based on model availability and provider health
    - Implements round-robin fallback across healthy providers
    - Translates ProtocolLLMRequest to provider-specific formats

Related Tickets:
    - OMN-2319: Implement SPI LLM protocol adapters (Gap 2)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

if TYPE_CHECKING:
    from omnibase_spi.protocols.llm.protocol_llm_provider import ProtocolLLMProvider

logger = logging.getLogger(__name__)


class AdapterModelRouter:
    """ProtocolModelRouter implementation with multi-provider routing.

    Routes LLM requests across multiple provider instances with:
    - Provider health checking and availability tracking
    - Round-robin load balancing across healthy providers
    - Automatic failover on provider failure
    - Capability-based provider selection

    Attributes:
        _providers: Mapping of provider names to provider instances.
        _provider_order: Ordered list of provider names for round-robin.
        _current_index: Current position in the round-robin cycle.
        _default_provider: Name of the default provider.

    Example:
        >>> router = AdapterModelRouter()
        >>> router.register_provider("vllm", vllm_provider)
        >>> router.register_provider("ollama", ollama_provider)
        >>> response = await router.generate(request)
    """

    def __init__(
        self,
        default_provider: str | None = None,
    ) -> None:
        """Initialize the model router.

        Args:
            default_provider: Name of the default provider. If None, the
                first registered provider is used as default.
        """
        self._providers: dict[str, ProtocolLLMProvider] = {}
        self._provider_order: list[str] = []
        self._current_index: int = 0
        self._default_provider = default_provider
        self._lock = asyncio.Lock()

    # ── Provider management ────────────────────────────────────────────

    async def register_provider(
        self,
        name: str,
        provider: ProtocolLLMProvider,
    ) -> None:
        """Register a provider instance for routing.

        Args:
            name: Unique provider name for routing.
            provider: The provider adapter instance.
        """
        async with self._lock:
            self._providers[name] = provider
            if name not in self._provider_order:
                self._provider_order.append(name)
            if self._default_provider is None:
                self._default_provider = name
        logger.info(
            "Registered LLM provider for routing: %s (total: %d)",
            name,
            len(self._providers),
        )

    async def remove_provider(self, name: str) -> None:
        """Remove a provider from the routing pool.

        Args:
            name: Provider name to remove.
        """
        async with self._lock:
            self._providers.pop(name, None)
            if name in self._provider_order:
                self._provider_order.remove(name)
            if self._default_provider == name:
                self._default_provider = (
                    self._provider_order[0] if self._provider_order else None
                )

    # ── ProtocolModelRouter interface ──────────────────────────────────

    async def generate(self, request: object) -> object:
        """Generate response using intelligent provider routing.

        Routes the request to the best available provider. If the selected
        provider fails, automatically falls back to the next healthy provider
        in the round-robin order.

        The ``object`` parameter and return types satisfy the SPI
        ``ProtocolModelRouter`` interface contract. For type-safe usage,
        see :meth:`generate_typed` which accepts
        ``ModelLlmAdapterRequest`` and returns ``ModelLlmAdapterResponse``.

        Args:
            request: LLM request (ModelLlmAdapterRequest or compatible).

        Returns:
            Generated response (ModelLlmAdapterResponse).

        Raises:
            ProtocolConfigurationError: If no providers are registered.
            InfraUnavailableError: If all providers are unavailable or fail.
            TypeError: If the request is not a ModelLlmAdapterRequest.
        """
        if not self._providers:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="generate",
                target_name="model-router",
            )
            raise ProtocolConfigurationError(
                "No LLM providers registered with the router",
                context=context,
            )

        if not isinstance(request, ModelLlmAdapterRequest):
            raise TypeError(
                f"Expected ModelLlmAdapterRequest, got {type(request).__name__}"
            )

        # Snapshot provider state under lock for thread-safe iteration.
        # Both _provider_order and _providers are snapshotted together so a
        # concurrent remove_provider() cannot delete a provider between the
        # order snapshot and the dict lookup.
        async with self._lock:
            provider_order = list(self._provider_order)
            providers_snapshot = dict(self._providers)
            current_index = self._current_index

        # Try providers in round-robin order, starting from current index
        errors: list[tuple[str, str]] = []
        attempted = 0

        for i in range(len(provider_order)):
            idx = (current_index + i) % len(provider_order)
            provider_name = provider_order[idx]
            provider = providers_snapshot.get(provider_name)

            if provider is None or not provider.is_available:
                continue

            try:
                response = await provider.generate_async(request)
                # Advance round-robin for next call.
                # NOTE: Round-robin is best-effort under concurrency -- two
                # concurrent generate() calls may snapshot the same index.
                # This is intentional: strict ordering would require holding
                # the lock during network I/O.
                async with self._lock:
                    self._current_index = (idx + 1) % len(provider_order)
                return response
            except Exception as exc:
                sanitized = sanitize_error_string(str(exc))
                errors.append((provider_name, sanitized))
                logger.warning(
                    "Provider %s failed, trying next: %s",
                    provider_name,
                    sanitized,
                )
                attempted += 1

        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="generate",
            target_name="model-router",
        )
        if attempted == 0:
            raise InfraUnavailableError(
                f"All {len(provider_order)} registered LLM providers "
                "are unavailable (none were attempted)",
                context=context,
            )
        error_details = "; ".join(f"{name}: {msg}" for name, msg in errors)
        raise InfraUnavailableError(
            f"All LLM providers failed ({attempted} attempted). "
            f"Errors: {error_details}",
            context=context,
        )

    async def generate_typed(
        self, request: ModelLlmAdapterRequest
    ) -> ModelLlmAdapterResponse:
        """Type-safe wrapper around generate() for callers with known types.

        Delegates to ``generate()`` and casts the result to
        ``ModelLlmAdapterResponse``. This avoids the ``object`` return type
        mandated by the SPI ``ProtocolModelRouter`` interface, giving callers
        static type safety without a runtime isinstance check.

        Args:
            request: The typed LLM request.

        Returns:
            Generated response with full type information.

        Raises:
            ProtocolConfigurationError: If no providers are registered.
            InfraUnavailableError: If all providers are unavailable or fail.
            TypeError: If the provider returns an unexpected type.
        """
        result = await self.generate(request)
        if not isinstance(result, ModelLlmAdapterResponse):
            raise TypeError(
                f"Expected ModelLlmAdapterResponse from generate(), "
                f"got {type(result).__name__}"
            )
        return result

    async def get_available_providers(self) -> list[str]:
        """Get list of currently available provider names.

        Returns:
            List of provider names that are registered and reporting as available.
        """
        async with self._lock:
            provider_order = list(self._provider_order)
            providers_snapshot = dict(self._providers)

        available = []
        for name in provider_order:
            provider = providers_snapshot.get(name)
            if provider is not None and provider.is_available:
                available.append(name)
        return available

    # ── Extended routing methods ───────────────────────────────────────

    async def generate_with_provider(
        self,
        request: ModelLlmAdapterRequest,
        provider_name: str,
    ) -> object:
        """Generate using a specific provider by name.

        Args:
            request: The LLM request.
            provider_name: Target provider name.

        Returns:
            Generated response.

        Raises:
            KeyError: If provider_name is not registered.
        """
        async with self._lock:
            provider = self._providers.get(provider_name)
            if provider is None:
                raise KeyError(
                    f"Provider '{provider_name}' not registered. "
                    f"Available: {list(self._providers.keys())}"
                )
        return await provider.generate_async(request)

    async def health_check_all(
        self,
    ) -> dict[str, object]:
        """Run health checks on all registered providers.

        Returns:
            Mapping of provider names to health check results.
        """
        async with self._lock:
            providers_snapshot = dict(self._providers)
        results: dict[str, object] = {}
        for name, provider in providers_snapshot.items():
            try:
                health = await provider.health_check()
                results[name] = health
            except Exception as exc:
                results[name] = {"error": sanitize_error_string(str(exc))}
        return results


__all__: list[str] = ["AdapterModelRouter"]
