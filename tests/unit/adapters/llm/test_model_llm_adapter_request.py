# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmAdapterRequest."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)


class TestModelLlmAdapterRequest:
    """Tests for the ModelLlmAdapterRequest Pydantic model."""

    def test_basic_request(self) -> None:
        """Basic request with required fields."""
        req = ModelLlmAdapterRequest(
            prompt="Hello, world!",
            model_name="qwen2.5-coder-14b",
        )
        assert req.prompt == "Hello, world!"
        assert req.model_name == "qwen2.5-coder-14b"
        assert req.parameters == {}
        assert req.max_tokens is None
        assert req.temperature is None

    def test_request_with_parameters(self) -> None:
        """Request with all optional fields."""
        req = ModelLlmAdapterRequest(
            prompt="Explain ONEX",
            model_name="gpt-4",
            parameters={"top_p": 0.9},
            max_tokens=500,
            temperature=0.7,
        )
        assert req.max_tokens == 500
        assert req.temperature == 0.7
        assert req.parameters == {"top_p": 0.9}

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        req = ModelLlmAdapterRequest(
            prompt="Hello",
            model_name="test",
        )
        with pytest.raises(Exception):
            req.prompt = "Changed"  # type: ignore[misc]

    def test_empty_prompt_rejected(self) -> None:
        """Empty prompt is rejected."""
        with pytest.raises(Exception):
            ModelLlmAdapterRequest(
                prompt="",
                model_name="test",
            )

    def test_temperature_bounds(self) -> None:
        """Temperature must be 0.0 to 2.0."""
        with pytest.raises(Exception):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="test",
                temperature=-0.1,
            )
        with pytest.raises(Exception):
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="test",
                temperature=2.1,
            )

    def test_default_parameters_is_empty_dict(self) -> None:
        """parameters defaults to an empty dict."""
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.parameters == {}
        assert isinstance(req.parameters, dict)

    def test_default_max_tokens_is_none(self) -> None:
        """max_tokens defaults to None (provider default)."""
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.max_tokens is None

    def test_default_temperature_is_none(self) -> None:
        """temperature defaults to None (provider default)."""
        req = ModelLlmAdapterRequest(prompt="Hello", model_name="test")
        assert req.temperature is None

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolLLMRequest."""
        req = ModelLlmAdapterRequest(
            prompt="Test",
            model_name="test-model",
            parameters={"key": "value"},
            max_tokens=100,
            temperature=0.5,
        )
        assert isinstance(req.prompt, str)
        assert isinstance(req.model_name, str)
        assert isinstance(req.parameters, dict)
