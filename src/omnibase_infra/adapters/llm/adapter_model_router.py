# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete ProtocolModelRouter implementation with multi-provider routing.

Coordinates request routing across multiple ProtocolLLMProvider instances
based on availability, capability matching, and round-robin load balancing.
Integrates with MixinAsyncCircuitBreaker for failover support.

Architecture:
    - Maintains a registry of named providers
    - Routes requests based on model availability and provider health
    - Implements round-robin fallback across healthy providers
    - Translates ProtocolLLMRequest to provider-specific formats

Related Tickets:
    - OMN-2319: Implement SPI LLM protocol adapters (Gap 2)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)

if TYPE_CHECKING:
    from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
        AdapterLlmProviderOpenai,
    )

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
        self._providers: dict[str, AdapterLlmProviderOpenai] = {}
        self._provider_order: list[str] = []
        self._current_index: int = 0
        self._default_provider = default_provider

    # ── Provider management ────────────────────────────────────────────

    def register_provider(
        self,
        name: str,
        provider: AdapterLlmProviderOpenai,
    ) -> None:
        """Register a provider instance for routing.

        Args:
            name: Unique provider name for routing.
            provider: The provider adapter instance.
        """
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

    def remove_provider(self, name: str) -> None:
        """Remove a provider from the routing pool.

        Args:
            name: Provider name to remove.
        """
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

        Args:
            request: LLM request (ModelLlmAdapterRequest or compatible).

        Returns:
            Generated response (ModelLlmAdapterResponse).

        Raises:
            RuntimeError: If no providers are registered or all are unavailable.
            TypeError: If the request is not a ModelLlmAdapterRequest.
        """
        if not self._providers:
            raise RuntimeError("No LLM providers registered with the router")

        if not isinstance(request, ModelLlmAdapterRequest):
            raise TypeError(
                f"Expected ModelLlmAdapterRequest, got {type(request).__name__}"
            )

        # Try providers in round-robin order, starting from current index
        errors: list[tuple[str, Exception]] = []
        attempted = 0

        for i in range(len(self._provider_order)):
            idx = (self._current_index + i) % len(self._provider_order)
            provider_name = self._provider_order[idx]
            provider = self._providers.get(provider_name)

            if provider is None or not provider.is_available:
                continue

            try:
                response = await provider.generate_async(request)
                # Advance round-robin for next call
                self._current_index = (idx + 1) % len(self._provider_order)
                return response
            except Exception as exc:
                errors.append((provider_name, exc))
                logger.warning(
                    "Provider %s failed, trying next: %s",
                    provider_name,
                    exc,
                )
                attempted += 1

        # All providers failed
        error_details = "; ".join(f"{name}: {exc}" for name, exc in errors)
        raise RuntimeError(
            f"All LLM providers failed ({attempted} attempted). Errors: {error_details}"
        )

    async def get_available_providers(self) -> list[str]:
        """Get list of currently available provider names.

        Returns:
            List of provider names that are registered and reporting as available.
        """
        available = []
        for name in self._provider_order:
            provider = self._providers.get(name)
            if provider is not None and provider.is_available:
                available.append(name)
        return available

    # ── Extended routing methods ───────────────────────────────────────

    async def generate_with_provider(
        self,
        request: ModelLlmAdapterRequest,
        provider_name: str,
    ) -> ModelLlmAdapterResponse:
        """Generate using a specific provider by name.

        Args:
            request: The LLM request.
            provider_name: Target provider name.

        Returns:
            Generated response.

        Raises:
            KeyError: If provider_name is not registered.
        """
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
        results: dict[str, object] = {}
        for name, provider in self._providers.items():
            try:
                health = await provider.health_check()
                results[name] = health
            except Exception as exc:
                results[name] = {"error": str(exc)}
        return results


__all__: list[str] = ["AdapterModelRouter"]
