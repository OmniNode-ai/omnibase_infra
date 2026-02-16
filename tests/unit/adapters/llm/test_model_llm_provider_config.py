# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmProviderConfig."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.model_llm_provider_config import (
    ModelLlmProviderConfig,
)


class TestModelLlmProviderConfig:
    """Tests for the ModelLlmProviderConfig Pydantic model."""

    def test_local_provider_config(self) -> None:
        """Local provider without API key."""
        config = ModelLlmProviderConfig(
            provider_name="openai-compatible",
            base_url="http://192.168.86.201:8000",
            default_model="qwen2.5-coder-14b",
            provider_type="local",
        )
        assert config.provider_name == "openai-compatible"
        assert config.api_key is None
        assert config.base_url == "http://192.168.86.201:8000"
        assert config.provider_type == "local"

    def test_external_provider_config(self) -> None:
        """External provider with API key."""
        config = ModelLlmProviderConfig(
            provider_name="openai",
            api_key="sk-test-key",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4",
            provider_type="external",
        )
        assert config.api_key == "sk-test-key"
        assert config.provider_type == "external"

    def test_defaults(self) -> None:
        """Default values."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.api_key is None
        assert config.base_url is None
        assert config.default_model == ""
        assert config.connection_timeout == 30
        assert config.max_retries == 3
        assert config.provider_type == "local"

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        config = ModelLlmProviderConfig(provider_name="test")
        with pytest.raises(Exception):
            config.provider_name = "other"  # type: ignore[misc]

    def test_timeout_bounds(self) -> None:
        """Connection timeout has valid bounds."""
        with pytest.raises(Exception):
            ModelLlmProviderConfig(
                provider_name="test",
                connection_timeout=0,
            )
        with pytest.raises(Exception):
            ModelLlmProviderConfig(
                provider_name="test",
                connection_timeout=601,
            )

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolProviderConfig."""
        config = ModelLlmProviderConfig(
            provider_name="test",
            base_url="http://localhost:8000",
            default_model="test-model",
        )
        assert isinstance(config.provider_name, str)
        assert isinstance(config.base_url, str)
        assert isinstance(config.default_model, str)
        assert isinstance(config.connection_timeout, int)
