# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterLlmToolProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from omnibase_infra.adapters.llm.adapter_llm_tool_provider import (
    AdapterLlmToolProvider,
)
from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter


def _make_mock_provider(name: str = "test") -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()
    type(provider).provider_name = PropertyMock(return_value=name)
    type(provider).is_available = PropertyMock(return_value=True)
    provider.close = AsyncMock()
    return provider


class TestAdapterLlmToolProviderRegistration:
    """Tests for provider registration."""

    def test_register_provider(self) -> None:
        """Register adds provider to tool provider and router."""
        tool_provider = AdapterLlmToolProvider()
        provider = _make_mock_provider("vllm")
        tool_provider.register_provider("vllm", provider)
        assert "vllm" in tool_provider._providers
        assert "vllm" in tool_provider._router._providers

    def test_list_providers(self) -> None:
        """List returns all registered provider names."""
        tool_provider = AdapterLlmToolProvider()
        tool_provider.register_provider("a", _make_mock_provider("a"))
        tool_provider.register_provider("b", _make_mock_provider("b"))
        assert tool_provider.list_providers() == ["a", "b"]


class TestAdapterLlmToolProviderAccess:
    """Tests for provider access methods."""

    @pytest.mark.asyncio
    async def test_get_model_router(self) -> None:
        """Returns the configured model router."""
        tool_provider = AdapterLlmToolProvider()
        router = await tool_provider.get_model_router()
        assert isinstance(router, AdapterModelRouter)

    @pytest.mark.asyncio
    async def test_get_openai_provider(self) -> None:
        """Returns registered OpenAI provider."""
        tool_provider = AdapterLlmToolProvider()
        mock = _make_mock_provider("openai")
        tool_provider.register_provider("openai", mock)
        provider = await tool_provider.get_openai_provider()
        assert provider is mock

    @pytest.mark.asyncio
    async def test_get_ollama_provider(self) -> None:
        """Returns registered Ollama provider."""
        tool_provider = AdapterLlmToolProvider()
        mock = _make_mock_provider("ollama")
        tool_provider.register_provider("ollama", mock)
        provider = await tool_provider.get_ollama_provider()
        assert provider is mock

    @pytest.mark.asyncio
    async def test_get_gemini_provider(self) -> None:
        """Returns registered Gemini provider."""
        tool_provider = AdapterLlmToolProvider()
        mock = _make_mock_provider("gemini")
        tool_provider.register_provider("gemini", mock)
        provider = await tool_provider.get_gemini_provider()
        assert provider is mock

    @pytest.mark.asyncio
    async def test_get_claude_provider(self) -> None:
        """Returns registered Claude provider."""
        tool_provider = AdapterLlmToolProvider()
        mock = _make_mock_provider("claude")
        tool_provider.register_provider("claude", mock)
        provider = await tool_provider.get_claude_provider()
        assert provider is mock

    @pytest.mark.asyncio
    async def test_get_missing_provider_raises(self) -> None:
        """Accessing unregistered provider raises KeyError."""
        tool_provider = AdapterLlmToolProvider()
        with pytest.raises(KeyError, match="not registered"):
            await tool_provider.get_openai_provider()

    def test_get_provider_by_name(self) -> None:
        """Get provider by arbitrary name."""
        tool_provider = AdapterLlmToolProvider()
        mock = _make_mock_provider("custom")
        tool_provider.register_provider("custom", mock)
        assert tool_provider.get_provider_by_name("custom") is mock

    def test_get_provider_by_name_missing(self) -> None:
        """Missing provider raises KeyError."""
        tool_provider = AdapterLlmToolProvider()
        with pytest.raises(KeyError):
            tool_provider.get_provider_by_name("nonexistent")


class TestAdapterLlmToolProviderLifecycle:
    """Tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        """Close all closes each provider."""
        tool_provider = AdapterLlmToolProvider()
        providers = [_make_mock_provider(f"p{i}") for i in range(3)]
        for i, p in enumerate(providers):
            tool_provider.register_provider(f"p{i}", p)

        await tool_provider.close_all()

        for p in providers:
            p.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_all_handles_errors(self) -> None:
        """Close all continues on individual errors."""
        tool_provider = AdapterLlmToolProvider()
        failing = _make_mock_provider("failing")
        failing.close = AsyncMock(side_effect=RuntimeError("close failed"))
        tool_provider.register_provider("failing", failing)

        working = _make_mock_provider("working")
        tool_provider.register_provider("working", working)

        # Should not raise
        await tool_provider.close_all()
        working.close.assert_awaited_once()
