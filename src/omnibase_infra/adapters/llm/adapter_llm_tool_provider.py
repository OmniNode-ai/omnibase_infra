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
    from omnibase_spi.protocols.llm.protocol_llm_provider import ProtocolLLMProvider

logger = logging.getLogger(__name__)


class AdapterLlmToolProvider:
    """ProtocolLLMToolProvider implementation for unified LLM access.

    Aggregates multiple LLM provider adapters and an intelligent model
    router, providing a single entry point for LLM services in the
    ONEX container DI system.

    Note:
        Getter methods (``get_model_router``, ``get_gemini_provider``, etc.)
        are ``async def`` despite performing no async work internally. This
        satisfies the SPI ``ProtocolLLMToolProvider`` async interface contract,
        allowing future implementations to perform async initialization or
        lazy provider setup without breaking the interface.

    Provider Registration Names:
        The typed getter methods (``get_gemini_provider``, etc.) expect
        providers to be registered under the canonical names defined by
        the ``PROVIDER_NAME_*`` class constants.  Callers **must** use
        these constants (or their string values) when calling
        ``register_provider`` to ensure the typed getters resolve
        correctly.

    Attributes:
        _router: The model router instance.
        _providers: Named provider instances.

    Example:
        >>> tool_provider = AdapterLlmToolProvider()
        >>> await tool_provider.register_provider("vllm", vllm_adapter)
        >>> router = await tool_provider.get_model_router()
        >>> response = await router.generate(request)
    """

    # Canonical registration names expected by the typed getter methods.
    PROVIDER_NAME_GEMINI: str = "gemini"
    PROVIDER_NAME_OPENAI: str = "openai"
    PROVIDER_NAME_OLLAMA: str = "ollama"
    PROVIDER_NAME_CLAUDE: str = "claude"

    def __init__(self) -> None:
        """Initialize the tool provider with an empty router."""
        self._router = AdapterModelRouter()
        self._providers: dict[str, ProtocolLLMProvider] = {}

    # ── Provider registration ──────────────────────────────────────────

    async def register_provider(
        self,
        name: str,
        provider: ProtocolLLMProvider,
    ) -> None:
        """Register a provider for access via the tool provider.

        The provider is also registered with the model router for
        automatic routing.

        Args:
            name: Unique provider name.
            provider: The provider adapter instance.
        """
        self._providers[name] = provider
        await self._router.register_provider(name, provider)
        logger.info("Registered LLM provider in tool provider: %s", name)

    # ── ProtocolLLMToolProvider interface ───────────────────────────────

    async def get_model_router(self) -> AdapterModelRouter:
        """Get configured model router with registered providers.

        Returns:
            The model router with all registered providers.
        """
        return self._router

    async def get_gemini_provider(self) -> ProtocolLLMProvider:
        """Get Gemini LLM provider instance.

        Expects the provider to be registered under
        ``PROVIDER_NAME_GEMINI`` (``"gemini"``).

        Returns:
            Configured Gemini provider.

        Raises:
            KeyError: If no provider is registered under the expected name.
        """
        return self._get_provider(self.PROVIDER_NAME_GEMINI)

    async def get_openai_provider(self) -> ProtocolLLMProvider:
        """Get OpenAI LLM provider instance.

        Expects the provider to be registered under
        ``PROVIDER_NAME_OPENAI`` (``"openai"``).

        Returns:
            Configured OpenAI provider.

        Raises:
            KeyError: If no provider is registered under the expected name.
        """
        return self._get_provider(self.PROVIDER_NAME_OPENAI)

    async def get_ollama_provider(self) -> ProtocolLLMProvider:
        """Get Ollama LLM provider instance.

        Expects the provider to be registered under
        ``PROVIDER_NAME_OLLAMA`` (``"ollama"``).

        Returns:
            Configured Ollama provider.

        Raises:
            KeyError: If no provider is registered under the expected name.
        """
        return self._get_provider(self.PROVIDER_NAME_OLLAMA)

    async def get_claude_provider(self) -> ProtocolLLMProvider:
        """Get Claude LLM provider instance (Anthropic).

        Expects the provider to be registered under
        ``PROVIDER_NAME_CLAUDE`` (``"claude"``).

        Returns:
            Configured Claude provider.

        Raises:
            KeyError: If no provider is registered under the expected name.
        """
        return self._get_provider(self.PROVIDER_NAME_CLAUDE)

    # ── Generic provider access ────────────────────────────────────────

    def get_provider_by_name(self, name: str) -> ProtocolLLMProvider:
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
        """Close all registered provider transports.

        Calls ``close()`` on each provider that exposes a close method.
        Providers that do not implement ``close()`` are silently skipped.
        """
        for name, provider in self._providers.items():
            close_fn = getattr(provider, "close", None)
            if close_fn is None:
                continue
            try:
                await close_fn()
                logger.debug("Closed LLM provider: %s", name)
            except Exception:
                logger.warning(
                    "Failed to close LLM provider: %s",
                    name,
                    exc_info=True,
                )

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_provider(self, name: str) -> ProtocolLLMProvider:
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
                f"Providers must be registered with the exact name "
                f"expected by the getter (see PROVIDER_NAME_* constants). "
                f"Available providers: {available}"
            )
        return provider


__all__: list[str] = ["AdapterLlmToolProvider"]
