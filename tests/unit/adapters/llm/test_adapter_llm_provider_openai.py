# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for AdapterLlmProviderOpenai."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    AdapterLlmProviderOpenai,
    _TransportHolder,
)
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_model_capabilities import (
    ModelLlmModelCapabilities,
)
from omnibase_infra.adapters.llm.model_llm_provider_config import (
    ModelLlmProviderConfig,
)


class TestAdapterLlmProviderOpenaiProperties:
    """Tests for provider properties."""

    def test_provider_name(self) -> None:
        """Provider name returns configured value."""
        adapter = AdapterLlmProviderOpenai(
            provider_name="vllm-coder",
        )
        assert adapter.provider_name == "vllm-coder"

    def test_provider_type_default(self) -> None:
        """Provider type defaults to 'local'."""
        adapter = AdapterLlmProviderOpenai()
        assert adapter.provider_type == "local"

    def test_is_available_default(self) -> None:
        """Provider starts as available."""
        adapter = AdapterLlmProviderOpenai()
        assert adapter.is_available is True

    def test_supports_streaming_v1(self) -> None:
        """Streaming is not supported in v1."""
        adapter = AdapterLlmProviderOpenai()
        assert adapter.supports_streaming() is False

    def test_supports_async(self) -> None:
        """Async is always supported."""
        adapter = AdapterLlmProviderOpenai()
        assert adapter.supports_async() is True


class TestAdapterLlmProviderOpenaiConfigure:
    """Tests for provider configuration."""

    def test_configure_updates_fields(self) -> None:
        """Configure updates base URL, API key, model, and type."""
        adapter = AdapterLlmProviderOpenai()
        config = ModelLlmProviderConfig(
            provider_name="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4",
            provider_type="external",
        )
        adapter.configure(config)
        assert adapter._base_url == "https://api.openai.com/v1"
        assert adapter._api_key == "sk-test"
        assert adapter._default_model == "gpt-4"
        assert adapter.provider_type == "external"


class TestAdapterLlmProviderOpenaiValidation:
    """Tests for request validation."""

    def test_validate_valid_request(self) -> None:
        """Valid request passes validation."""
        adapter = AdapterLlmProviderOpenai()
        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="qwen2.5-coder-14b",
        )
        assert adapter.validate_request(request) is True


class TestAdapterLlmProviderOpenaiCostEstimation:
    """Tests for cost estimation."""

    def test_local_provider_zero_cost(self) -> None:
        """Local providers always estimate zero cost."""
        adapter = AdapterLlmProviderOpenai(provider_type="local")
        request = ModelLlmAdapterRequest(
            prompt="Hello world",
            model_name="qwen2.5-coder-14b",
        )
        assert adapter.estimate_cost(request) == 0.0

    def test_external_provider_with_capabilities(self) -> None:
        """External provider estimates cost from capabilities."""
        caps = ModelLlmModelCapabilities(
            model_name="gpt-4",
            cost_per_1k_input_tokens=0.03,
            cost_per_1k_output_tokens=0.06,
        )
        adapter = AdapterLlmProviderOpenai(
            provider_type="external",
            model_capabilities={"gpt-4": caps},
        )
        request = ModelLlmAdapterRequest(
            prompt="Hello world, this is a test prompt",
            model_name="gpt-4",
            max_tokens=100,
        )
        cost = adapter.estimate_cost(request)
        assert cost > 0.0

    def test_external_provider_unknown_model(self) -> None:
        """Unknown model returns zero cost."""
        adapter = AdapterLlmProviderOpenai(provider_type="external")
        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="unknown-model",
        )
        assert adapter.estimate_cost(request) == 0.0


class TestAdapterLlmProviderOpenaiGenerate:
    """Tests for generation methods."""

    @pytest.mark.asyncio
    async def test_generate_stream_raises(self) -> None:
        """Synchronous streaming raises NotImplementedError."""
        adapter = AdapterLlmProviderOpenai()
        request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test",
        )
        with pytest.raises(NotImplementedError):
            adapter.generate_stream(request)

    @pytest.mark.asyncio
    async def test_get_provider_info(self) -> None:
        """Provider info returns expected fields."""
        adapter = AdapterLlmProviderOpenai(
            base_url="http://localhost:8000",
            default_model="test-model",
            provider_name="test-provider",
        )
        info = await adapter.get_provider_info()
        assert isinstance(info, dict)
        assert info["name"] == "test-provider"
        assert info["base_url"] == "http://localhost:8000"
        assert info["default_model"] == "test-model"


class TestAdapterLlmProviderOpenaiTranslation:
    """Tests for request translation."""

    def test_translate_request_maps_fields(self) -> None:
        """Translation correctly maps SPI fields to infra fields."""
        adapter = AdapterLlmProviderOpenai(
            base_url="http://localhost:8000",
            default_model="qwen2.5-coder-14b",
        )
        spi_request = ModelLlmAdapterRequest(
            prompt="Explain ONEX",
            model_name="qwen2.5-coder-14b",
            temperature=0.7,
            max_tokens=500,
        )
        infra_request = adapter._translate_request(spi_request)

        assert infra_request.base_url == "http://localhost:8000"
        assert infra_request.model == "qwen2.5-coder-14b"
        assert infra_request.temperature == 0.7
        assert infra_request.max_tokens == 500
        assert len(infra_request.messages) == 1
        assert infra_request.messages[0]["role"] == "user"
        assert infra_request.messages[0]["content"] == "Explain ONEX"

    def test_translate_request_uses_request_model(self) -> None:
        """Translation uses the model_name from the request directly."""
        adapter = AdapterLlmProviderOpenai(
            base_url="http://localhost:8000",
            default_model="qwen2.5-coder-14b",
        )
        spi_request = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="other-model",
        )
        infra_request = adapter._translate_request(spi_request)
        assert infra_request.model == "other-model"

    def test_default_model_fallback_unreachable(self) -> None:
        """Document that model_name fallback to default_model is unreachable.

        ModelLlmAdapterRequest.model_name has min_length=1, so Pydantic
        rejects empty strings before the adapter's ``or self._default_model``
        fallback in _translate_request can ever execute. This test documents
        that the fallback path is dead code by construction.
        """
        with pytest.raises(Exception):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="",
            )


class TestTransportHolder:
    """Tests for the internal transport holder."""

    def test_transport_holder_init(self) -> None:
        """Transport holder initializes with LLM HTTP transport."""
        holder = _TransportHolder(target_name="test")
        assert holder._llm_target_name == "test"
