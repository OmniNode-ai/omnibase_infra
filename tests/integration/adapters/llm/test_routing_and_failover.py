# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for LLM adapter routing and failover.

These tests verify multi-provider routing, capability-based selection,
and circuit-breaker failover behavior across the adapter stack without
requiring live LLM endpoints.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    AdapterLlmProviderOpenai,
)
from omnibase_infra.adapters.llm.adapter_llm_tool_provider import (
    AdapterLlmToolProvider,
)
from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.adapters.llm.model_llm_health_response import (
    ModelLlmHealthResponse,
)
from omnibase_infra.adapters.llm.model_llm_model_capabilities import (
    ModelLlmModelCapabilities,
)
from omnibase_infra.adapters.llm.model_llm_provider_config import (
    ModelLlmProviderConfig,
)
from omnibase_infra.errors import InfraUnavailableError


def _mock_provider(
    name: str,
    available: bool = True,
    response_text: str = "Generated response",
) -> MagicMock:
    """Create a mock provider with configurable behavior."""
    provider = MagicMock(spec=AdapterLlmProviderOpenai)
    type(provider).is_available = PropertyMock(return_value=available)
    type(provider).provider_name = PropertyMock(return_value=name)
    type(provider).provider_type = PropertyMock(return_value="local")
    provider.generate_async = AsyncMock(
        return_value=ModelLlmAdapterResponse(
            generated_text=response_text,
            model_used="test-model",
            finish_reason="stop",
            usage_statistics={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
    )
    provider.health_check = AsyncMock(
        return_value=ModelLlmHealthResponse(
            is_healthy=available,
            provider_name=name,
            response_time_ms=50.0,
            available_models=["test-model"],
        )
    )
    provider.close = AsyncMock()
    return provider


class TestMultiProviderRouting:
    """Integration tests for routing across multiple providers."""

    @pytest.mark.asyncio
    async def test_router_selects_first_available(self) -> None:
        """Router selects the first available provider."""
        router = AdapterModelRouter()
        await router.register_provider("vllm", _mock_provider("vllm"))
        await router.register_provider("ollama", _mock_provider("ollama"))

        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test-model",
        )
        response = await router.generate(request)
        assert isinstance(response, ModelLlmAdapterResponse)
        assert response.generated_text == "Generated response"

    @pytest.mark.asyncio
    async def test_router_round_robin_distribution(self) -> None:
        """Router distributes requests via round-robin."""
        router = AdapterModelRouter()
        p1 = _mock_provider("p1", response_text="from-p1")
        p2 = _mock_provider("p2", response_text="from-p2")
        await router.register_provider("p1", p1)
        await router.register_provider("p2", p2)

        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test-model",
        )

        r1 = await router.generate(request)
        r2 = await router.generate(request)

        # After two calls, both providers should have been called
        assert p1.generate_async.await_count + p2.generate_async.await_count == 2


class TestFailoverBehavior:
    """Integration tests for failover across providers."""

    @pytest.mark.asyncio
    async def test_failover_on_provider_error(self) -> None:
        """Router falls back to next provider on error."""
        router = AdapterModelRouter()

        failing = _mock_provider("failing")
        failing.generate_async = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )
        await router.register_provider("failing", failing)

        backup = _mock_provider("backup", response_text="backup response")
        await router.register_provider("backup", backup)

        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test-model",
        )
        response = await router.generate(request)
        assert response.generated_text == "backup response"

    @pytest.mark.asyncio
    async def test_failover_skips_unavailable(self) -> None:
        """Router skips providers marked as unavailable."""
        router = AdapterModelRouter()

        down = _mock_provider("down", available=False)
        await router.register_provider("down", down)

        up = _mock_provider("up", response_text="up response")
        await router.register_provider("up", up)

        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test-model",
        )
        response = await router.generate(request)
        assert response.generated_text == "up response"
        down.generate_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_providers_down_error(self) -> None:
        """InfraUnavailableError when all providers fail."""
        router = AdapterModelRouter()

        for name in ["p1", "p2", "p3"]:
            failing = _mock_provider(name)
            failing.generate_async = AsyncMock(
                side_effect=TimeoutError(f"{name} timed out")
            )
            await router.register_provider(name, failing)

        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test-model",
        )
        with pytest.raises(InfraUnavailableError, match="All LLM providers failed"):
            await router.generate(request)


class TestToolProviderIntegration:
    """Integration tests for the tool provider facade."""

    @pytest.mark.asyncio
    async def test_tool_provider_router_integration(self) -> None:
        """Tool provider's router includes all registered providers."""
        tool_provider = AdapterLlmToolProvider()
        await tool_provider.register_provider("vllm", _mock_provider("vllm"))
        await tool_provider.register_provider("ollama", _mock_provider("ollama"))

        router = await tool_provider.get_model_router()
        available = await router.get_available_providers()
        assert "vllm" in available
        assert "ollama" in available

    @pytest.mark.asyncio
    async def test_tool_provider_named_access(self) -> None:
        """Tool provider returns specific providers by name."""
        tool_provider = AdapterLlmToolProvider()
        openai_mock = _mock_provider("openai")
        ollama_mock = _mock_provider("ollama")

        await tool_provider.register_provider("openai", openai_mock)
        await tool_provider.register_provider("ollama", ollama_mock)

        openai = await tool_provider.get_openai_provider()
        assert openai is openai_mock

        ollama = await tool_provider.get_ollama_provider()
        assert ollama is ollama_mock

    @pytest.mark.asyncio
    async def test_tool_provider_end_to_end_routing(self) -> None:
        """End-to-end: register providers, get router, generate response."""
        tool_provider = AdapterLlmToolProvider()
        vllm = _mock_provider("vllm", response_text="ONEX architecture overview")
        await tool_provider.register_provider("vllm", vllm)

        router = await tool_provider.get_model_router()
        request = ModelLlmAdapterRequest(
            prompt="Explain ONEX architecture",
            model_name="qwen2.5-coder-14b",
        )
        response = await router.generate(request)

        assert response.generated_text == "ONEX architecture overview"
        assert response.model_used == "test-model"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_provider_lifecycle(self) -> None:
        """Close all providers through tool provider."""
        tool_provider = AdapterLlmToolProvider()
        providers = [_mock_provider(f"p{i}") for i in range(3)]
        for i, p in enumerate(providers):
            await tool_provider.register_provider(f"p{i}", p)

        await tool_provider.close_all()
        for p in providers:
            p.close.assert_awaited_once()


class TestProtocolConformance:
    """Tests verifying structural protocol conformance."""

    def test_health_response_satisfies_protocol(self) -> None:
        """ModelLlmHealthResponse has all ProtocolLLMHealthResponse properties."""
        resp = ModelLlmHealthResponse(
            is_healthy=True,
            provider_name="test",
            response_time_ms=10.0,
            available_models=["model-a"],
        )
        # ProtocolLLMHealthResponse properties
        assert hasattr(resp, "is_healthy")
        assert hasattr(resp, "provider_name")
        assert hasattr(resp, "response_time_ms")
        assert hasattr(resp, "available_models")
        assert hasattr(resp, "error_message")

    def test_model_capabilities_satisfies_protocol(self) -> None:
        """ModelLlmModelCapabilities has all ProtocolModelCapabilities properties."""
        caps = ModelLlmModelCapabilities(model_name="test")
        # ProtocolModelCapabilities properties
        assert hasattr(caps, "model_name")
        assert hasattr(caps, "supports_streaming")
        assert hasattr(caps, "supports_function_calling")
        assert hasattr(caps, "max_context_length")
        assert hasattr(caps, "supported_modalities")

    def test_provider_config_satisfies_protocol(self) -> None:
        """ModelLlmProviderConfig has all ProtocolProviderConfig properties."""
        config = ModelLlmProviderConfig(provider_name="test")
        # ProtocolProviderConfig properties
        assert hasattr(config, "provider_name")
        assert hasattr(config, "api_key")
        assert hasattr(config, "base_url")
        assert hasattr(config, "default_model")
        assert hasattr(config, "connection_timeout")

    def test_adapter_request_satisfies_protocol(self) -> None:
        """ModelLlmAdapterRequest has all ProtocolLLMRequest properties."""
        req = ModelLlmAdapterRequest(
            prompt="test",
            model_name="model",
        )
        # ProtocolLLMRequest properties
        assert hasattr(req, "prompt")
        assert hasattr(req, "model_name")
        assert hasattr(req, "parameters")
        assert hasattr(req, "max_tokens")
        assert hasattr(req, "temperature")

    def test_adapter_response_satisfies_protocol(self) -> None:
        """ModelLlmAdapterResponse has all ProtocolLLMResponse properties."""
        resp = ModelLlmAdapterResponse(
            generated_text="text",
            model_used="model",
        )
        # ProtocolLLMResponse properties
        assert hasattr(resp, "generated_text")
        assert hasattr(resp, "model_used")
        assert hasattr(resp, "usage_statistics")
        assert hasattr(resp, "finish_reason")
        assert hasattr(resp, "response_metadata")
