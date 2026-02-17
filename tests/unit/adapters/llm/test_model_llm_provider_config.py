# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmProviderConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.adapters.llm.model_llm_provider_config import (
    ModelLlmProviderConfig,
)
from omnibase_spi.protocols.types.protocol_llm_types import ProtocolProviderConfig


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
        """Default values for all optional fields."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.api_key is None
        assert config.base_url is None
        assert config.default_model == ""
        assert config.connection_timeout == 30
        assert config.max_retries == 3
        assert config.provider_type == "local"

    def test_default_api_key_is_none(self) -> None:
        """api_key defaults to None (no authentication required)."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.api_key is None

    def test_default_base_url_is_none(self) -> None:
        """base_url defaults to None (must be configured before use)."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.base_url is None

    def test_default_model_is_empty_string(self) -> None:
        """default_model defaults to empty string, not None."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.default_model == ""
        assert isinstance(config.default_model, str)

    def test_default_connection_timeout_is_30(self) -> None:
        """connection_timeout defaults to 30 seconds."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.connection_timeout == 30

    def test_default_max_retries_is_3(self) -> None:
        """max_retries defaults to 3 attempts."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.max_retries == 3

    def test_default_provider_type_is_local(self) -> None:
        """provider_type defaults to 'local'."""
        config = ModelLlmProviderConfig(provider_name="test")
        assert config.provider_type == "local"

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        config = ModelLlmProviderConfig(provider_name="test")
        with pytest.raises(ValidationError):
            config.provider_name = "other"  # type: ignore[misc]

    def test_timeout_bounds(self) -> None:
        """Connection timeout has valid bounds."""
        with pytest.raises(ValidationError):
            ModelLlmProviderConfig(
                provider_name="test",
                connection_timeout=0,
            )
        with pytest.raises(ValidationError):
            ModelLlmProviderConfig(
                provider_name="test",
                connection_timeout=601,
            )

    def test_invalid_provider_type_rejected(self) -> None:
        """Invalid provider_type values are rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            ModelLlmProviderConfig(
                provider_name="test",
                provider_type="invalid",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "provider_type",
        ["local", "external_trusted", "external"],
    )
    def test_valid_provider_types_accepted(self, provider_type: str) -> None:
        """All valid provider_type literal values are accepted."""
        config = ModelLlmProviderConfig(
            provider_name="test",
            provider_type=provider_type,  # type: ignore[arg-type]
        )
        assert config.provider_type == provider_type

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolProviderConfig."""
        config = ModelLlmProviderConfig(
            provider_name="test",
            base_url="http://localhost:8000",
            default_model="test-model",
        )
        assert isinstance(config, ProtocolProviderConfig)
        assert isinstance(config.provider_name, str)
        assert isinstance(config.base_url, str)
        assert isinstance(config.default_model, str)
        assert isinstance(config.connection_timeout, int)
