# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ModelLlmAdapterResponse."""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)


class TestModelLlmAdapterResponse:
    """Tests for the ModelLlmAdapterResponse Pydantic model."""

    def test_basic_response(self) -> None:
        """Basic response with generated text."""
        resp = ModelLlmAdapterResponse(
            generated_text="Hello, world!",
            model_used="qwen2.5-coder-14b",
            finish_reason="stop",
        )
        assert resp.generated_text == "Hello, world!"
        assert resp.model_used == "qwen2.5-coder-14b"
        assert resp.finish_reason == "stop"
        assert resp.usage_statistics == {}
        assert resp.response_metadata == {}

    def test_content_alias(self) -> None:
        """Content property returns generated_text."""
        resp = ModelLlmAdapterResponse(
            generated_text="Test content",
            model_used="test-model",
        )
        assert resp.content == "Test content"

    def test_response_with_usage(self) -> None:
        """Response with usage statistics."""
        resp = ModelLlmAdapterResponse(
            generated_text="Hello",
            model_used="test-model",
            usage_statistics={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
        assert resp.usage_statistics["prompt_tokens"] == 10

    def test_frozen_model(self) -> None:
        """Model is immutable."""
        resp = ModelLlmAdapterResponse(
            generated_text="Hello",
            model_used="test",
        )
        with pytest.raises(Exception):
            resp.generated_text = "Changed"  # type: ignore[misc]

    def test_satisfies_protocol(self) -> None:
        """Verify structural compatibility with ProtocolLLMResponse."""
        resp = ModelLlmAdapterResponse(
            generated_text="text",
            model_used="model",
            usage_statistics={"key": "value"},
            finish_reason="stop",
            response_metadata={"latency": 100},
        )
        assert isinstance(resp.generated_text, str)
        assert isinstance(resp.model_used, str)
        assert isinstance(resp.usage_statistics, dict)
        assert isinstance(resp.finish_reason, str)
        assert isinstance(resp.response_metadata, dict)
