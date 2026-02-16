# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Concrete ProtocolLLMToolProvider implementation.

Provides centralized access to LLM providers and the model router,
implementing the SPI ProtocolLLMToolProvider interface for container-based
dependency injection.

Architecture:
    - Holds references to named provider instances
    - Creates and manages the AdapterModelRouter
    - Provides accessor methods for specific provider backends
    - Supports lazy provider initialization

Related Tickets:
    - OMN-2319: Implement SPI LLM protocol adapters (Gap 3)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter

if TYPE_CHECKING:
    from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
        AdapterLlmProviderOpenai,
    )

logger = logging.getLogger(__name__)


class AdapterLlmToolProvider:
    """ProtocolLLMToolProvider implementation for unified LLM access.

    Aggregates multiple LLM provider adapters and an intelligent model
    router, providing a single entry point for LLM services in the
    ONEX container DI system.

    Attributes:
        _router: The model router instance.
        _providers: Named provider instances.

    Example:
        >>> tool_provider = AdapterLlmToolProvider()
        >>> tool_provider.register_provider("vllm", vllm_adapter)
        >>> router = await tool_provider.get_model_router()
        >>> response = await router.generate(request)
    """

    def __init__(self) -> None:
        """Initialize the tool provider with an empty router."""
        self._router = AdapterModelRouter()
        self._providers: dict[str, AdapterLlmProviderOpenai] = {}

    # ── Provider registration ──────────────────────────────────────────

    def register_provider(
        self,
        name: str,
        provider: AdapterLlmProviderOpenai,
    ) -> None:
        """Register a provider for access via the tool provider.

        The provider is also registered with the model router for
        automatic routing.

        Args:
            name: Unique provider name.
            provider: The provider adapter instance.
        """
        self._providers[name] = provider
        self._router.register_provider(name, provider)
        logger.info("Registered LLM provider in tool provider: %s", name)

    # ── ProtocolLLMToolProvider interface ───────────────────────────────

    async def get_model_router(self) -> AdapterModelRouter:
        """Get configured model router with registered providers.

        Returns:
            The model router with all registered providers.
        """
        return self._router

    async def get_gemini_provider(self) -> AdapterLlmProviderOpenai:
        """Get Gemini LLM provider instance.

        Returns:
            Configured Gemini provider.

        Raises:
            KeyError: If no Gemini provider is registered.
        """
        return self._get_provider("gemini")

    async def get_openai_provider(self) -> AdapterLlmProviderOpenai:
        """Get OpenAI LLM provider instance.

        Returns:
            Configured OpenAI provider.

        Raises:
            KeyError: If no OpenAI provider is registered.
        """
        return self._get_provider("openai")

    async def get_ollama_provider(self) -> AdapterLlmProviderOpenai:
        """Get Ollama LLM provider instance.

        Returns:
            Configured Ollama provider.

        Raises:
            KeyError: If no Ollama provider is registered.
        """
        return self._get_provider("ollama")

    async def get_claude_provider(self) -> AdapterLlmProviderOpenai:
        """Get Claude LLM provider instance (Anthropic).

        Returns:
            Configured Claude provider.

        Raises:
            KeyError: If no Claude provider is registered.
        """
        return self._get_provider("claude")

    # ── Generic provider access ────────────────────────────────────────

    def get_provider_by_name(self, name: str) -> AdapterLlmProviderOpenai:
        """Get a provider by its registered name.

        Args:
            name: Registered provider name.

        Returns:
            The provider adapter instance.

        Raises:
            KeyError: If the provider is not registered.
        """
        return self._get_provider(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of registered provider names.
        """
        return list(self._providers.keys())

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def close_all(self) -> None:
        """Close all registered provider transports."""
        for name, provider in self._providers.items():
            try:
                await provider.close()
                logger.debug("Closed LLM provider: %s", name)
            except Exception:
                logger.warning(
                    "Failed to close LLM provider: %s",
                    name,
                    exc_info=True,
                )

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_provider(self, name: str) -> AdapterLlmProviderOpenai:
        """Get a provider by name with error handling.

        Args:
            name: Provider name to look up.

        Returns:
            The provider instance.

        Raises:
            KeyError: If the provider is not registered.
        """
        provider = self._providers.get(name)
        if provider is None:
            available = list(self._providers.keys())
            raise KeyError(
                f"LLM provider '{name}' is not registered. "
                f"Available providers: {available}"
            )
        return provider


__all__: list[str] = ["AdapterLlmToolProvider"]
