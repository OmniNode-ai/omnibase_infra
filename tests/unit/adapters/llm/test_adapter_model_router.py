# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterModelRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.errors import InfraUnavailableError, ProtocolConfigurationError


def _make_mock_provider(
    name: str = "test",
    available: bool = True,
) -> MagicMock:
    """Create a mock provider for testing."""
    provider = MagicMock()
    type(provider).is_available = PropertyMock(return_value=available)
    type(provider).provider_name = PropertyMock(return_value=name)
    provider.generate_async = AsyncMock(
        return_value=ModelLlmAdapterResponse(
            generated_text=f"Response from {name}",
            model_used="test-model",
            finish_reason="stop",
        )
    )
    provider.health_check = AsyncMock()
    return provider


def _make_request() -> ModelLlmAdapterRequest:
    """Create a test request."""
    return ModelLlmAdapterRequest(
        prompt="Hello",
        model_name="test-model",
    )


class TestAdapterModelRouterRegistration:
    """Tests for provider registration."""

    @pytest.mark.asyncio
    async def test_register_provider(self) -> None:
        """Register adds provider to routing pool."""
        router = AdapterModelRouter()
        provider = _make_mock_provider("vllm")
        await router.register_provider("vllm", provider)
        available = await router.get_available_providers()
        assert "vllm" in available

    @pytest.mark.asyncio
    async def test_register_sets_default(self) -> None:
        """First registered provider becomes default and is routable."""
        router = AdapterModelRouter()
        provider = _make_mock_provider("vllm")
        await router.register_provider("vllm", provider)
        # Verify it is routable by generating a request
        response = await router.generate(_make_request())
        assert isinstance(response, ModelLlmAdapterResponse)
        provider.generate_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remove_provider(self) -> None:
        """Remove provider from routing pool."""
        router = AdapterModelRouter()
        provider = _make_mock_provider("vllm")
        await router.register_provider("vllm", provider)
        await router.remove_provider("vllm")
        available = await router.get_available_providers()
        assert "vllm" not in available

    @pytest.mark.asyncio
    async def test_remove_default_updates(self) -> None:
        """Removing default provider selects next available."""
        router = AdapterModelRouter()
        await router.register_provider("a", _make_mock_provider("a"))
        b_provider = _make_mock_provider("b")
        await router.register_provider("b", b_provider)
        await router.remove_provider("a")
        # After removing "a", "b" should be the only available provider
        available = await router.get_available_providers()
        assert available == ["b"]
        # Verify "b" is actually used for generation
        response = await router.generate(_make_request())
        assert isinstance(response, ModelLlmAdapterResponse)
        b_provider.generate_async.assert_awaited_once()


class TestAdapterModelRouterGenerate:
    """Tests for request generation routing."""

    @pytest.mark.asyncio
    async def test_generate_routes_to_available(self) -> None:
        """Generate routes to the first available provider."""
        router = AdapterModelRouter()
        provider = _make_mock_provider("vllm")
        await router.register_provider("vllm", provider)

        response = await router.generate(_make_request())
        assert isinstance(response, ModelLlmAdapterResponse)
        provider.generate_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_no_providers_raises(self) -> None:
        """Generate with no providers raises ProtocolConfigurationError."""
        router = AdapterModelRouter()
        with pytest.raises(
            ProtocolConfigurationError, match="No LLM providers registered"
        ):
            await router.generate(_make_request())

    @pytest.mark.asyncio
    async def test_generate_wrong_type_raises(self) -> None:
        """Generate with wrong request type raises TypeError."""
        router = AdapterModelRouter()
        await router.register_provider("vllm", _make_mock_provider("vllm"))
        with pytest.raises(TypeError, match="Expected ModelLlmAdapterRequest"):
            await router.generate("not a request")

    @pytest.mark.asyncio
    async def test_generate_failover(self) -> None:
        """Generate fails over to next provider on error."""
        router = AdapterModelRouter()

        failing = _make_mock_provider("failing")
        failing.generate_async = AsyncMock(side_effect=ConnectionError("down"))
        await router.register_provider("failing", failing)

        working = _make_mock_provider("working")
        await router.register_provider("working", working)

        response = await router.generate(_make_request())
        assert isinstance(response, ModelLlmAdapterResponse)
        assert response.generated_text == "Response from working"

    @pytest.mark.asyncio
    async def test_generate_all_fail_raises(self) -> None:
        """Generate raises when all providers fail."""
        router = AdapterModelRouter()

        for name in ["a", "b"]:
            provider = _make_mock_provider(name)
            provider.generate_async = AsyncMock(
                side_effect=ConnectionError(f"{name} down")
            )
            await router.register_provider(name, provider)

        with pytest.raises(InfraUnavailableError, match="All LLM providers failed"):
            await router.generate(_make_request())

    @pytest.mark.asyncio
    async def test_generate_skips_unavailable(self) -> None:
        """Generate skips unavailable providers."""
        router = AdapterModelRouter()

        unavailable = _make_mock_provider("unavailable", available=False)
        await router.register_provider("unavailable", unavailable)

        available = _make_mock_provider("available")
        await router.register_provider("available", available)

        response = await router.generate(_make_request())
        assert isinstance(response, ModelLlmAdapterResponse)
        unavailable.generate_async.assert_not_awaited()
        available.generate_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_round_robin_advances(self) -> None:
        """Round-robin alternates which provider handles each request."""
        router = AdapterModelRouter()
        provider_a = _make_mock_provider("a")
        provider_b = _make_mock_provider("b")
        await router.register_provider("a", provider_a)
        await router.register_provider("b", provider_b)

        # First call should go to provider "a"
        resp1 = await router.generate(_make_request())
        assert resp1.generated_text == "Response from a"
        provider_a.generate_async.assert_awaited_once()
        provider_b.generate_async.assert_not_awaited()

        # Second call should go to provider "b"
        resp2 = await router.generate(_make_request())
        assert resp2.generated_text == "Response from b"
        provider_b.generate_async.assert_awaited_once()

        # Third call wraps around back to provider "a"
        resp3 = await router.generate(_make_request())
        assert resp3.generated_text == "Response from a"
        assert provider_a.generate_async.await_count == 2

    @pytest.mark.asyncio
    async def test_generate_all_providers_unavailable(self) -> None:
        """Generate raises when all providers are registered but unavailable."""
        router = AdapterModelRouter()
        await router.register_provider(
            "offline_a", _make_mock_provider("offline_a", available=False)
        )
        await router.register_provider(
            "offline_b", _make_mock_provider("offline_b", available=False)
        )

        with pytest.raises(InfraUnavailableError, match="unavailable"):
            await router.generate(_make_request())

    @pytest.mark.asyncio
    async def test_generate_all_unavailable_zero_attempted(self) -> None:
        """All-unavailable results in zero attempted providers."""
        router = AdapterModelRouter()

        providers: list[MagicMock] = []
        for name in ["x", "y", "z"]:
            provider = _make_mock_provider(name, available=False)
            await router.register_provider(name, provider)
            providers.append(provider)

        with pytest.raises(InfraUnavailableError, match="none were attempted"):
            await router.generate(_make_request())

        # None of the providers should have been called
        for provider in providers:
            provider.generate_async.assert_not_awaited()


class TestAdapterModelRouterAvailability:
    """Tests for availability queries."""

    @pytest.mark.asyncio
    async def test_get_available_providers(self) -> None:
        """Returns only available providers."""
        router = AdapterModelRouter()
        await router.register_provider("a", _make_mock_provider("a", available=True))
        await router.register_provider("b", _make_mock_provider("b", available=False))
        await router.register_provider("c", _make_mock_provider("c", available=True))

        available = await router.get_available_providers()
        assert available == ["a", "c"]

    @pytest.mark.asyncio
    async def test_generate_with_provider(self) -> None:
        """Direct provider generation works."""
        router = AdapterModelRouter()
        provider = _make_mock_provider("vllm")
        await router.register_provider("vllm", provider)

        response = await router.generate_with_provider(_make_request(), "vllm")
        assert isinstance(response, ModelLlmAdapterResponse)

    @pytest.mark.asyncio
    async def test_generate_with_unknown_provider(self) -> None:
        """Unknown provider name raises KeyError."""
        router = AdapterModelRouter()
        with pytest.raises(KeyError, match="not registered"):
            await router.generate_with_provider(_make_request(), "unknown")

    @pytest.mark.asyncio
    async def test_health_check_all(self) -> None:
        """Health check runs on all registered providers."""
        router = AdapterModelRouter()
        await router.register_provider("a", _make_mock_provider("a"))
        await router.register_provider("b", _make_mock_provider("b"))

        results = await router.health_check_all()
        assert "a" in results
        assert "b" in results
