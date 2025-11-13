#!/usr/bin/env python3
"""
Unit tests for LLM Effect models.

Tests Pydantic model validation, defaults, and edge cases.
"""

from uuid import UUID, uuid4

import pytest

from src.omninode_bridge.nodes.llm_effect.v1_0_0.models import (
    EnumLLMTier,
    ModelLLMConfig,
    ModelLLMRequest,
    ModelLLMResponse,
)


class TestEnumLLMTier:
    """Test cases for EnumLLMTier."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert EnumLLMTier.LOCAL.value == "LOCAL"
        assert EnumLLMTier.CLOUD_FAST.value == "CLOUD_FAST"
        assert EnumLLMTier.CLOUD_PREMIUM.value == "CLOUD_PREMIUM"

    def test_tier_from_string(self):
        """Test creating tier from string."""
        assert EnumLLMTier("CLOUD_FAST") == EnumLLMTier.CLOUD_FAST
        assert EnumLLMTier("LOCAL") == EnumLLMTier.LOCAL


class TestModelLLMRequest:
    """Test cases for ModelLLMRequest."""

    def test_minimal_request(self):
        """Test creating request with minimal required fields."""
        request = ModelLLMRequest(prompt="Generate code")

        assert request.prompt == "Generate code"
        assert request.tier == EnumLLMTier.CLOUD_FAST  # Default
        assert request.max_tokens == 4000  # Default
        assert request.temperature == 0.7  # Default
        assert isinstance(request.execution_id, UUID)

    def test_full_request(self):
        """Test creating request with all fields."""
        correlation_id = uuid4()
        execution_id = uuid4()

        request = ModelLLMRequest(
            prompt="Generate Python function",
            system_prompt="You are a coding assistant",
            tier=EnumLLMTier.CLOUD_FAST,
            model_override="glm-4.5",
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            operation_type="node_generation",
            context_window="# Existing code...",
            enable_streaming=False,
            max_retries=2,
            timeout_seconds=30.0,
            metadata={"project": "test"},
            correlation_id=correlation_id,
            execution_id=execution_id,
        )

        assert request.prompt == "Generate Python function"
        assert request.system_prompt == "You are a coding assistant"
        assert request.tier == EnumLLMTier.CLOUD_FAST
        assert request.model_override == "glm-4.5"
        assert request.max_tokens == 2000
        assert request.temperature == 0.5
        assert request.correlation_id == correlation_id
        assert request.execution_id == execution_id

    def test_prompt_validation(self):
        """Test prompt length validation."""
        # Empty prompt should fail
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMRequest(prompt="")

        # Very long prompt (within limit) should succeed
        long_prompt = "a" * 50000
        request = ModelLLMRequest(prompt=long_prompt)
        assert len(request.prompt) == 50000

    def test_temperature_validation(self):
        """Test temperature range validation."""
        # Valid range
        request = ModelLLMRequest(prompt="test", temperature=0.0)
        assert request.temperature == 0.0

        request = ModelLLMRequest(prompt="test", temperature=2.0)
        assert request.temperature == 2.0

        # Invalid range
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMRequest(prompt="test", temperature=-0.1)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMRequest(prompt="test", temperature=2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid range
        request = ModelLLMRequest(prompt="test", max_tokens=1)
        assert request.max_tokens == 1

        request = ModelLLMRequest(prompt="test", max_tokens=128000)
        assert request.max_tokens == 128000

        # Invalid range
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMRequest(prompt="test", max_tokens=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMRequest(prompt="test", max_tokens=128001)


class TestModelLLMResponse:
    """Test cases for ModelLLMResponse."""

    def test_minimal_response(self):
        """Test creating response with minimal required fields."""
        response = ModelLLMResponse(
            generated_text="def fibonacci(n): ...",
            model_used="glm-4.5",
            tier_used=EnumLLMTier.CLOUD_FAST,
            tokens_input=25,
            tokens_output=45,
            tokens_total=70,
            latency_ms=1250.5,
            cost_usd=0.000014,
            finish_reason="stop",
        )

        assert response.generated_text == "def fibonacci(n): ..."
        assert response.model_used == "glm-4.5"
        assert response.tier_used == EnumLLMTier.CLOUD_FAST
        assert response.tokens_total == 70
        assert response.latency_ms == 1250.5
        assert response.cost_usd == 0.000014
        assert response.finish_reason == "stop"
        assert response.truncated is False  # Default
        assert response.retry_count == 0  # Default

    def test_truncated_response(self):
        """Test response with truncation."""
        response = ModelLLMResponse(
            generated_text="Truncated...",
            model_used="glm-4.5",
            tier_used=EnumLLMTier.CLOUD_FAST,
            tokens_input=100,
            tokens_output=4000,
            tokens_total=4100,
            latency_ms=5000.0,
            cost_usd=0.00082,
            finish_reason="length",
            truncated=True,
            warnings=["Response truncated at max_tokens limit"],
        )

        assert response.finish_reason == "length"
        assert response.truncated is True
        assert len(response.warnings) == 1
        assert "truncated" in response.warnings[0].lower()

    def test_token_validation(self):
        """Test token count validation (non-negative)."""
        # Valid tokens
        response = ModelLLMResponse(
            generated_text="test",
            model_used="glm-4.5",
            tier_used=EnumLLMTier.CLOUD_FAST,
            tokens_input=0,
            tokens_output=0,
            tokens_total=0,
            latency_ms=100.0,
            cost_usd=0.0,
            finish_reason="stop",
        )
        assert response.tokens_total == 0

        # Negative tokens should fail
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMResponse(
                generated_text="test",
                model_used="glm-4.5",
                tier_used=EnumLLMTier.CLOUD_FAST,
                tokens_input=-1,
                tokens_output=10,
                tokens_total=9,
                latency_ms=100.0,
                cost_usd=0.0,
                finish_reason="stop",
            )


class TestModelLLMConfig:
    """Test cases for ModelLLMConfig."""

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = ModelLLMConfig(zai_api_key="test_key_12345")

        assert config.zai_api_key == "test_key_12345"
        assert config.zai_base_url == "https://api.z.ai/api/anthropic"  # Default
        assert config.default_model_cloud_fast == "glm-4.5"  # Default
        assert config.context_window_cloud_fast == 128000  # Default
        assert config.circuit_breaker_threshold == 5  # Default

    def test_full_config(self):
        """Test creating config with all fields."""
        config = ModelLLMConfig(
            zai_api_key="test_key_12345",
            zai_base_url="https://custom.api/v1",
            ollama_base_url="http://localhost:11434",
            default_model_local="llama-3.1-8b",
            default_model_cloud_fast="glm-4.5",
            default_model_cloud_premium="glm-4.6",
            context_window_local=8192,
            context_window_cloud_fast=128000,
            context_window_cloud_premium=128000,
            cost_per_1m_input_cloud_fast=0.15,
            cost_per_1m_output_cloud_fast=0.15,
            circuit_breaker_threshold=10,
            circuit_breaker_timeout_seconds=120.0,
            http_timeout_seconds=90.0,
            max_retry_attempts=5,
        )

        assert config.zai_api_key == "test_key_12345"
        assert config.zai_base_url == "https://custom.api/v1"
        assert config.circuit_breaker_threshold == 10
        assert config.max_retry_attempts == 5

    def test_circuit_breaker_validation(self):
        """Test circuit breaker threshold validation."""
        # Valid range
        config = ModelLLMConfig(zai_api_key="test", circuit_breaker_threshold=1)
        assert config.circuit_breaker_threshold == 1

        config = ModelLLMConfig(zai_api_key="test", circuit_breaker_threshold=20)
        assert config.circuit_breaker_threshold == 20

        # Invalid range
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMConfig(zai_api_key="test", circuit_breaker_threshold=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMConfig(zai_api_key="test", circuit_breaker_threshold=21)

    def test_retry_validation(self):
        """Test retry attempts validation."""
        # Valid range
        config = ModelLLMConfig(zai_api_key="test", max_retry_attempts=1)
        assert config.max_retry_attempts == 1

        config = ModelLLMConfig(zai_api_key="test", max_retry_attempts=5)
        assert config.max_retry_attempts == 5

        # Invalid range
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMConfig(zai_api_key="test", max_retry_attempts=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelLLMConfig(zai_api_key="test", max_retry_attempts=6)
